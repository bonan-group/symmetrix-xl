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
#include <vector>

// TODO: remove some of these headers?
#include "KokkosBatched_Util.hpp"
#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemm_Decl.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#include "Kokkos_Sort.hpp"
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
#include "neighbor_graph_kokkos.hpp"
#include "device_backend.hpp"
#include "factorized_blas.hpp"

using Kokkos::ALL;
using Kokkos::LayoutRight;
using Kokkos::make_pair;
using Kokkos::MemoryUnmanaged;
using Kokkos::parallel_for;
using Kokkos::PerTeam;
using Kokkos::subview;
using Kokkos::TeamPolicy;
using Kokkos::TeamVectorRange;
using Kokkos::TeamVectorMDRange;
using Kokkos::View;
#include "mace_kokkos_kernel_launch_detail.hpp"

namespace {

struct DualLayerSourceEdge {
    int source;
    int local_edge;
};

struct DualLayerSourceEdgeLess {
    KOKKOS_FUNCTION
    bool operator()(
        const DualLayerSourceEdge left,
        const DualLayerSourceEdge right) const
    {
        if (left.source != right.source)
            return left.source < right.source;
        return left.local_edge < right.local_edge;
    }
};

std::size_t checked_workspace_product(
    const std::string_view name,
    const std::initializer_list<std::size_t> factors)
{
    std::size_t product = 1;
    for (const auto factor : factors) {
        if (factor != 0
            && product > std::numeric_limits<std::size_t>::max()/factor)
            throw std::length_error(
                "Single-layer "+std::string(name)+" size overflows size_t.");
        product *= factor;
    }
    return product;
}

std::size_t checked_workspace_sum(
    const std::string_view name,
    const std::initializer_list<std::size_t> terms)
{
    std::size_t total = 0;
    for (const auto term : terms) {
        if (total > std::numeric_limits<std::size_t>::max()-term)
            throw std::length_error(
                "Single-layer "+std::string(name)+" size overflows size_t.");
        total += term;
    }
    return total;
}

std::size_t checked_workspace_align_up(
    const std::string_view name,
    const std::size_t value,
    const std::size_t alignment)
{
    if (alignment == 0)
        throw std::logic_error("Single-layer workspace alignment is zero.");
    const std::size_t remainder = value%alignment;
    return remainder == 0 ? value : checked_workspace_sum(
        name, {value, alignment-remainder});
}

template <typename ViewType, typename ArenaType, typename... Extents>
ViewType make_tracked_workspace_alias(
    const ArenaType& arena,
    typename ViewType::pointer_type data,
    Extents... extents)
{
    (void)arena;
    return ViewType(
        data, typename ViewType::array_layout(extents...));
}

void copy_int_prefix(
    const Kokkos::DefaultExecutionSpace& execution_space,
    const Kokkos::View<int*>& storage,
    Kokkos::View<int*>& active,
    const std::vector<int>& values)
{
    if (values.empty()) {
        active = Kokkos::View<int*>();
        return;
    }
    if (storage.extent(0) < values.size())
        throw std::logic_error("Factorized graph storage does not cover its active prefix.");
    active = Kokkos::subview(
        storage,
        Kokkos::make_pair(std::size_t(0), values.size()));
    auto host = Kokkos::create_mirror_view(active);
    std::copy(values.begin(), values.end(), host.data());
    Kokkos::deep_copy(execution_space, active, host);
}

} // namespace

template <typename Precision>
std::string MACEKokkos<Precision>::streamed_edges_mode() const
{
    return mace_streamed_edges_mode_name(streamed_edges);
}

template <typename Precision>
void MACEKokkos<Precision>::set_execution_plan_request(
    std::string algorithm, std::string profile, std::string debug_plan)
{
    const auto requested = parse_mace_streamed_edges_mode(algorithm);
    set_streamed_edges(std::move(algorithm));
    execution_plan_.requested_algorithm = requested;
    execution_plan_.requested_profile =
        symmetrix::execution::parse_execution_profile(profile);
    if (!debug_plan.empty()
        && debug_plan != "mh0-direct-speed"
        && debug_plan != "mh0-direct-capacity-retained"
        && debug_plan != "mh0-direct-capacity-y-only"
        && debug_plan != "mh0-single-layer-tiled-v1"
        && debug_plan != "mh0-dual-layer-tiled-v1")
        throw std::invalid_argument(
            "_debug_execution_plan must be one of 'mh0-direct-speed', "
            "'mh0-direct-capacity-retained', or "
            "'mh0-direct-capacity-y-only', or "
            "'mh0-single-layer-tiled-v1', or "
            "'mh0-dual-layer-tiled-v1'.");
    if (!debug_plan.empty() && !mace_uses_direct_execution(requested))
        throw std::invalid_argument(
            "_debug_execution_plan requires streamed_edges='direct'.");
    if (debug_plan == "mh0-direct-speed"
        && execution_plan_.requested_profile
            != symmetrix::execution::ExecutionProfile::speed)
        throw std::invalid_argument(
            "mh0-direct-speed requires execution_profile='speed'.");
    if ((debug_plan == "mh0-direct-capacity-retained"
         || debug_plan == "mh0-direct-capacity-y-only"
         || debug_plan == "mh0-single-layer-tiled-v1"
         || debug_plan == "mh0-dual-layer-tiled-v1")
        && execution_plan_.requested_profile
            != symmetrix::execution::ExecutionProfile::capacity)
        throw std::invalid_argument(
            "MH-0 capacity debug plans require execution_profile='capacity'.");
    execution_plan_debug_id_ = std::move(debug_plan);
    execution_plan_.selection_source = execution_plan_debug_id_.empty()
        ? "native_request" : "debug_override";
    execution_plan_.state = "pending";
    execution_plan_.selected_id.clear();
    execution_plan_.selection_reason = "awaiting direct artifact resolution";
    execution_plan_.available_bytes = 0;
    execution_plan_.reserve_bytes = 0;
    execution_plan_.boundary_attempt = false;
    execution_plan_.candidates.clear();
}

template <typename Precision>
void MACEKokkos<Precision>::resolve_execution_plan()
{
    if (execution_plan_.requested_algorithm != streamed_edges)
        throw std::logic_error(
            "execution-plan algorithm differs from the active streamed-edge mode.");
    if (!mace_uses_direct_execution(streamed_edges)) {
        execution_plan_.state = "active";
        execution_plan_.selected_id = streamed_edges == MACEStreamedEdgesMode::generic
            ? "generic" : "legacy-materialized";
        execution_plan_.selection_reason = "explicit compatibility algorithm";
        return;
    }
    if (execution_plan_.requested_profile
        == symmetrix::execution::ExecutionProfile::capacity) {
        set_low_memory(true);
        execution_plan_.selection_reason =
            "capacity request pending prepared-graph memory selection";
        return;
    }
    set_low_memory(false);
    apply_preferred_speed_policy();
    execution_plan_.state = "active";
    execution_plan_.selected_id = "mh0-direct-speed";
    execution_plan_.selection_reason =
        execution_plan_debug_id_.empty()
        ? "speed request selected the backend-qualified throughput plan"
        : "debug override selected the backend-qualified throughput plan";
}

template <typename Precision>
const symmetrix::execution::ExecutionPlanReport&
MACEKokkos<Precision>::execution_plan_report() const
{
    return execution_plan_;
}

template <typename Precision>
void MACEKokkos<Precision>::set_streamed_edges(std::string mode)
{
    const auto requested = parse_mace_streamed_edges_mode(mode);
    if (requested != MACEStreamedEdgesMode::materialized && !supports_streamed_edges())
        throw std::invalid_argument(
            "Streamed edges require a format-v2 compact MACE or MACEField model.");
    if (mace_uses_prepared_execution(requested) && !supports_factorized())
        throw std::invalid_argument(
            "direct execution requires a factorable format-v2 "
            "compact MACE model.");
    Kokkos::fence();
    invalidate_all_interactions_prepared_graph();
    streamed_edges = requested;
    if (mace_uses_direct_execution(streamed_edges)
        && (jit_host_plugin_ready() || jit_device_plugin_ready()))
        factorized_source_strategy = FactorizedSourceStrategy::jit_plugin;
    if (use_m0_module())
        release_m0_polynomial_workspace();
    standard_r0_module_active = false;
    standard_r0_density_scale_fused = false;
    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges)) {
        R1 = decltype(R1)();
        R1_deriv = decltype(R1_deriv)();
    }
    if (mace_uses_prepared_execution(streamed_edges)) {
        active_types.clear();
        factorized_ready = false;
        factorized_schedule_dirty = true;
        invalidate_factorized_prepared_graph();
        Phi1r = decltype(Phi1r)();
        Phi1 = decltype(Phi1)();
        dPhi1r = decltype(dPhi1r)();
        dPhi1 = decltype(dPhi1)();
    }
    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges)) {
        R0 = decltype(R0)();
        R0_deriv = decltype(R0_deriv)();
    }
    if (mace_uses_prepared_execution(streamed_edges)) {
        A0_spline_values = decltype(A0_spline_values)();
        A0_spline_derivs = decltype(A0_spline_derivs)();
    }
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_streamed_edge_schedule(
    const int num_nodes,
    const int num_edges,
    Kokkos::View<const int*> num_neigh)
{
    if (streamed_first_neigh.extent(0) != num_nodes)
        Kokkos::realloc(streamed_first_neigh, num_nodes);
    auto first_neigh = streamed_first_neigh;
    Kokkos::parallel_scan(
        "MACEKokkos::prepare_streamed_edge_schedule",
        num_nodes,
        KOKKOS_LAMBDA (const int i, int& update, const bool final) {
            if (final)
                first_neigh(i) = update;
            update += num_neigh(i);
        });
    Kokkos::fence();
    all_interactions_schedule_build_count += 1;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    if (streamed_edges != MACEStreamedEdgesMode::generic
        && !supports_fused_streamed_reverse()
        && num_edges > streamed_edge_owned_limit) {
        streamed_edge_receivers = Kokkos::View<int*>();
        return;
    }
    if (streamed_edge_receivers.extent(0) != num_edges)
        Kokkos::realloc(streamed_edge_receivers, num_edges);
    auto edge_receivers = streamed_edge_receivers;
    Kokkos::parallel_for(
        "MACEKokkos::prepare_streamed_edge_receivers",
        Kokkos::TeamPolicy<>(num_nodes, Kokkos::AUTO),
        KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type team_member) {
            const int i = team_member.league_rank();
            const int i0 = first_neigh(i);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team_member, num_neigh(i)),
                [=] (const int j) {
                    edge_receivers(i0+j) = i;
                });
        });
    Kokkos::fence();
#else
    static_cast<void>(num_edges);
#endif
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_schedule(
    const int num_nodes,
    const int num_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices)
{
    prepare_factorized_schedule(
        num_nodes, num_nodes, num_edges, num_neigh, neigh_indices);
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_schedule(
    const int num_receivers,
    const int num_feature_nodes,
    const int num_edges,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices)
{
    if (static_cast<int>(num_neigh.extent(0)) != num_receivers
        || static_cast<int>(neigh_indices.extent(0)) != num_edges)
        throw std::invalid_argument("Execution R1 graph extents are inconsistent.");

    factorized_topology_validation_count += 1;
    const auto h_num_neigh =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), num_neigh);
    const auto h_neigh_indices =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), neigh_indices);
    auto schedule_num_neigh = std::vector<int>(num_receivers);
    auto schedule_neigh_indices = std::vector<int>(num_edges);
    for (int receiver=0; receiver<num_receivers; ++receiver)
        schedule_num_neigh[receiver] = h_num_neigh(receiver);
    for (int edge=0; edge<num_edges; ++edge)
        schedule_neigh_indices[edge] = h_neigh_indices(edge);

    if (factorized_prepared_graph_generation != 0
        && (schedule_num_neigh != execution_prepared_num_neigh_host
            || schedule_neigh_indices != execution_prepared_neigh_indices_host))
        invalidate_factorized_prepared_graph();

    if (prepare_factorized_schedule_host(
            num_receivers, num_feature_nodes,
            schedule_num_neigh, schedule_neigh_indices)) {
        factorized_execution_space.fence("Execution R1 fallback graph preparation");
        factorized_preparation_fence_count += 1;
    }
}

template <typename Precision>
void MACEKokkos<Precision>::invalidate_factorized_operator_benchmark()
{
    factorized_operator_benchmark_ready = false;
    factorized_operator_benchmark_token = 0;
    factorized_operator_benchmark_graph_generation = 0;
    factorized_operator_benchmark_evaluation_epoch = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::begin_factorized_production_evaluation()
{
    if (mace_uses_direct_execution(streamed_edges)) {
        if (!single_layer_readout
            && !jit_host_plugin_ready() && !jit_device_plugin_ready())
            throw std::runtime_error(
                "streamed_edges='direct' requires a loaded RTC artifact; "
                "direct execution does not fall back to another algorithm.");
        if (factorized_observer_enabled || execution_parameter_gradients_enabled)
            throw std::invalid_argument(
                "streamed_edges='direct' does not support Execution observers "
                "or parameter gradients; these features are unavailable.");
    }
    invalidate_factorized_operator_benchmark();
    factorized_completed_evaluation_epoch = 0;
    factorized_completed_evaluation_graph_generation = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::complete_factorized_prepared_evaluation(
    const std::uint64_t graph_generation)
{
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::logic_error(
            "Cannot complete a Execution R1 evaluation for a stale prepared graph.");
    factorized_completed_evaluation_epoch_counter += 1;
    if (factorized_completed_evaluation_epoch_counter == 0)
        factorized_completed_evaluation_epoch_counter += 1;
    factorized_completed_evaluation_epoch =
        factorized_completed_evaluation_epoch_counter;
    factorized_completed_evaluation_graph_generation = graph_generation;
}

template <typename Precision>
void MACEKokkos<Precision>::invalidate_all_interactions_prepared_graph()
{
    all_interactions_prepared_graph_generation = 0;
    all_interactions_prepared_geometry_graph_generation = 0;
    all_interactions_completed_graph_generation = 0;
    all_interactions_active_edge_count = 0;
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_all_interactions_graph(
    const int num_nodes,
    const std::span<const int> node_types,
    const std::span<const int> num_neigh,
    const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    if (streamed_edges != MACEStreamedEdgesMode::generic)
        throw std::invalid_argument(
            "Prepared streamed graphs require streamed_edges='generic'.");
    if (num_nodes < 0
        || node_types.size() != static_cast<std::size_t>(num_nodes)
        || num_neigh.size() != static_cast<std::size_t>(num_nodes)
        || neigh_indices.size() != neigh_types.size())
        throw std::invalid_argument(
            "Prepared streamed graph extents are inconsistent.");
    validate_graph_cardinality(
        static_cast<std::size_t>(num_nodes),
        static_cast<std::size_t>(num_nodes),
        neigh_indices.size());

    const auto prepared_node_types = std::vector<int>(
        node_types.begin(), node_types.end());
    const auto prepared_num_neigh = std::vector<int>(
        num_neigh.begin(), num_neigh.end());
    const auto prepared_neigh_indices = std::vector<int>(
        neigh_indices.begin(), neigh_indices.end());
    const auto prepared_neigh_types = std::vector<int>(
        neigh_types.begin(), neigh_types.end());
    std::size_t counted_edges = 0;
    for (int receiver=0; receiver<num_nodes; ++receiver) {
        if (prepared_num_neigh[receiver] < 0)
            throw std::invalid_argument(
                "Prepared streamed receiver degree is negative.");
        counted_edges += static_cast<std::size_t>(
            prepared_num_neigh[receiver]);
        if (prepared_node_types[receiver] < 0
            || prepared_node_types[receiver]
                >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Prepared streamed node type is out of range.");
    }
    if (counted_edges != prepared_neigh_indices.size())
        throw std::invalid_argument(
            "Prepared streamed receiver degrees do not sum to the edge count.");
    for (std::size_t edge=0; edge<prepared_neigh_indices.size(); ++edge) {
        if (prepared_neigh_indices[edge] < 0
            || prepared_neigh_indices[edge] >= num_nodes)
            throw std::out_of_range(
                "Prepared streamed source index is out of range.");
        if (prepared_neigh_types[edge] < 0
            || prepared_neigh_types[edge]
                >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Prepared streamed neighbor type is out of range.");
    }

    const bool same_identity = all_interactions_prepared_graph_generation != 0
        && prepared_node_types == all_interactions_node_types_host
        && prepared_num_neigh == all_interactions_num_neigh_host
        && prepared_neigh_indices == all_interactions_neigh_indices_host
        && prepared_neigh_types == all_interactions_neigh_types_host;
    if (same_identity)
        return all_interactions_prepared_graph_generation;

    invalidate_all_interactions_prepared_graph();
    std::vector<int> candidate_first_neigh(num_nodes);
    std::vector<int> candidate_edge_receivers(prepared_neigh_indices.size());
    int edge_offset = 0;
    for (int receiver=0; receiver<num_nodes; ++receiver) {
        candidate_first_neigh[receiver] = edge_offset;
        for (int offset=0; offset<prepared_num_neigh[receiver]; ++offset)
            candidate_edge_receivers[edge_offset+offset] = receiver;
        edge_offset += prepared_num_neigh[receiver];
    }

    all_interactions_candidate_node_types = toKokkosView(
        "Streamed all candidate node types", prepared_node_types);
    all_interactions_candidate_num_neigh = toKokkosView(
        "Streamed all candidate receiver degrees", prepared_num_neigh);
    all_interactions_candidate_first_neigh = toKokkosView(
        "Streamed all candidate receiver offsets", candidate_first_neigh);
    all_interactions_candidate_edge_receivers = toKokkosView(
        "Streamed all candidate receivers", candidate_edge_receivers);
    all_interactions_candidate_neigh_indices = toKokkosView(
        "Streamed all candidate source indices", prepared_neigh_indices);
    all_interactions_candidate_neigh_types = toKokkosView(
        "Streamed all candidate neighbor types", prepared_neigh_types);

    const int num_edges = static_cast<int>(prepared_neigh_indices.size());
    if (all_interactions_capacity_nodes < num_nodes) {
        all_interactions_active_num_neigh = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all active receiver degrees"),
            num_nodes);
        all_interactions_positions = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Streamed all positions"),
            3*static_cast<std::size_t>(num_nodes));
        all_interactions_displacements = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Streamed all displacements"),
            3*static_cast<std::size_t>(num_nodes));
        all_interactions_capacity_nodes = num_nodes;
    }
    if (all_interactions_capacity_edges < num_edges) {
        all_interactions_candidate_active = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Streamed all active flags"),
            num_edges);
        all_interactions_active_neigh_indices = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all active source indices"),
            num_edges);
        all_interactions_active_edge_receivers = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all active edge receivers"),
            num_edges);
        all_interactions_active_neigh_types = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all active neighbor types"),
            num_edges);
        all_interactions_candidate_xyz = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all candidate displacements"),
            3*static_cast<std::size_t>(num_edges));
        all_interactions_candidate_r = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all candidate distances"),
            num_edges);
        all_interactions_active_xyz = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing,
                "Streamed all active displacements"),
            3*static_cast<std::size_t>(num_edges));
        all_interactions_active_r = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Streamed all active distances"),
            num_edges);
        all_interactions_capacity_edges = num_edges;
    }
    if (all_interactions_geometry_invalid.extent(0) != 1)
        all_interactions_geometry_invalid = Kokkos::View<int*>(
            "Streamed all geometry invalid", 1);
    Kokkos::fence("Streamed all explicit graph preparation");

    all_interactions_node_types_host = prepared_node_types;
    all_interactions_num_neigh_host = prepared_num_neigh;
    all_interactions_neigh_indices_host = prepared_neigh_indices;
    all_interactions_neigh_types_host = prepared_neigh_types;
    all_interactions_graph_generation_counter += 1;
    if (all_interactions_graph_generation_counter == 0)
        all_interactions_graph_generation_counter += 1;
    all_interactions_prepared_graph_generation =
        all_interactions_graph_generation_counter;
    all_interactions_prepared_graph_count += 1;
    return all_interactions_prepared_graph_generation;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_all_interactions_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> reference_positions,
    const std::span<const double> reference_xyz,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::span<const int> pbc)
{
    if (graph_generation == 0
        || graph_generation != all_interactions_prepared_graph_generation)
        throw std::invalid_argument(
            "Streamed all geometry requires the current prepared graph token.");
    const std::size_t num_nodes = all_interactions_candidate_node_types.extent(0);
    const std::size_t num_edges =
        all_interactions_candidate_neigh_indices.extent(0);
    if (reference_positions.size() != 3*num_nodes
        || reference_xyz.size() != 3*num_edges
        || cell.size() != 9 || inverse_cell.size() != 9 || pbc.size() != 3)
        throw std::invalid_argument(
            "Streamed all prepared geometry extents are inconsistent.");
    const auto finite = [] (const double value) { return std::isfinite(value); };
    if (!std::all_of(reference_positions.begin(), reference_positions.end(), finite)
        || !std::all_of(reference_xyz.begin(), reference_xyz.end(), finite)
        || !std::all_of(cell.begin(), cell.end(), finite)
        || !std::all_of(inverse_cell.begin(), inverse_cell.end(), finite))
        throw std::invalid_argument(
            "Streamed all prepared geometry contains a non-finite value.");

    const auto copy_double = [&] (
        const char* label,
        const std::span<const double> source,
        Kokkos::View<double*>& destination) {
        const auto host = Kokkos::View<
            const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                source.data(), source.size());
        destination = Kokkos::View<double*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, std::string(label)),
            source.size());
        Kokkos::deep_copy(destination, host);
    };
    copy_double(
        "Streamed all reference positions", reference_positions,
        all_interactions_reference_positions);
    copy_double(
        "Streamed all reference displacements", reference_xyz,
        all_interactions_reference_xyz);
    copy_double("Streamed all cell", cell, all_interactions_cell);
    copy_double(
        "Streamed all inverse cell", inverse_cell,
        all_interactions_inverse_cell);
    const auto pbc_host = Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            pbc.data(), pbc.size());
    all_interactions_pbc = Kokkos::View<int*>(
        Kokkos::view_alloc(
            Kokkos::WithoutInitializing, "Streamed all pbc"),
        pbc.size());
    Kokkos::deep_copy(all_interactions_pbc, pbc_host);
    Kokkos::fence("Streamed all geometry preparation");
    all_interactions_prepared_geometry_graph_generation = graph_generation;
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_all_interactions_positions(
    const std::uint64_t graph_generation,
    const std::span<const double> positions)
{
    if (streamed_edges != MACEStreamedEdgesMode::generic)
        throw std::invalid_argument(
            "Prepared streamed positions require streamed_edges='generic'.");
    if (graph_generation == 0
        || graph_generation != all_interactions_prepared_graph_generation
        || graph_generation
            != all_interactions_prepared_geometry_graph_generation)
        throw std::invalid_argument(
            "Streamed all positions require current graph and geometry tokens.");
    const int num_nodes =
        static_cast<int>(all_interactions_candidate_node_types.extent(0));
    const int num_candidates =
        static_cast<int>(all_interactions_candidate_neigh_indices.extent(0));
    if (positions.size() != 3*static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "Streamed all positions do not match the graph extent.");
    if (!std::all_of(
            positions.begin(), positions.end(),
            [] (const double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "Streamed all positions contain a non-finite value.");

    const auto positions_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            positions.data(), positions.size());
    const auto current_positions = Kokkos::subview(
        all_interactions_positions,
        Kokkos::make_pair(std::size_t(0), positions.size()));
    Kokkos::deep_copy(current_positions, positions_host);
    const auto active_num_neigh = Kokkos::subview(
        all_interactions_active_num_neigh,
        Kokkos::make_pair(std::size_t(0), static_cast<std::size_t>(num_nodes)));
    Kokkos::deep_copy(active_num_neigh, 0);
    Kokkos::deep_copy(all_interactions_geometry_invalid, 0);

    const auto reference_positions = all_interactions_reference_positions;
    const auto reference_xyz = all_interactions_reference_xyz;
    const auto cell = all_interactions_cell;
    const auto inverse_cell = all_interactions_inverse_cell;
    const auto pbc = all_interactions_pbc;
    auto displacements = all_interactions_displacements;
    Kokkos::parallel_for(
        "MACEKokkos::all_interactions_atom_displacements",
        num_nodes,
        KOKKOS_LAMBDA (const int atom) {
            double delta[3];
            double fractional[3];
            for (int component=0; component<3; ++component)
                delta[component] = current_positions(3*atom+component)
                    -reference_positions(3*atom+component);
            if (pbc(0) == 0 && pbc(1) == 0 && pbc(2) == 0) {
                for (int component=0; component<3; ++component)
                    displacements(3*atom+component) = delta[component];
                return;
            }
            for (int lattice=0; lattice<3; ++lattice) {
                fractional[lattice] = 0.0;
                for (int component=0; component<3; ++component)
                    fractional[lattice] +=
                        delta[component]*inverse_cell(3*component+lattice);
                if (pbc(lattice) != 0)
                    fractional[lattice] -= Kokkos::floor(
                        fractional[lattice]+0.5);
            }
            for (int component=0; component<3; ++component) {
                double value = 0.0;
                for (int lattice=0; lattice<3; ++lattice)
                    value += fractional[lattice]*cell(3*lattice+component);
                displacements(3*atom+component) = value;
            }
        });

    const auto candidate_receivers = all_interactions_candidate_edge_receivers;
    const auto candidate_geometry_sources =
        all_interactions_candidate_neigh_indices;
    // Keep the skin topology fixed but compact the exact active graph in
    // receiver order for the generic streamed evaluator.
    auto candidate_active = all_interactions_candidate_active;
    auto candidate_xyz = all_interactions_candidate_xyz;
    auto candidate_r = all_interactions_candidate_r;
    auto invalid = all_interactions_geometry_invalid;
    const double cutoff = r_cut;
    Kokkos::parallel_for(
        "MACEKokkos::all_interactions_candidate_geometry",
        num_candidates,
        KOKKOS_LAMBDA (const int edge) {
            const int receiver = candidate_receivers(edge);
            const int source = candidate_geometry_sources(edge);
            double vector[3];
            double squared_distance = 0.0;
            for (int component=0; component<3; ++component) {
                vector[component] = reference_xyz(3*edge+component)
                    +displacements(3*source+component)
                    -displacements(3*receiver+component);
                squared_distance += vector[component]*vector[component];
            }
            if (!Kokkos::isfinite(squared_distance)
                || !(squared_distance > 0.0)) {
                candidate_active(edge) = 0;
                Kokkos::atomic_add(&invalid(0), 1);
                return;
            }
            const double distance = Kokkos::sqrt(squared_distance);
            for (int component=0; component<3; ++component)
                candidate_xyz(3*edge+component) = vector[component];
            candidate_r(edge) = distance;
            const int active = distance < cutoff ? 1 : 0;
            candidate_active(edge) = active;
            if (active != 0)
                Kokkos::atomic_add(&active_num_neigh(receiver), 1);
        });

    int active_edges = 0;
    Kokkos::parallel_reduce(
        "MACEKokkos::all_interactions_active_edge_count",
        num_nodes,
        KOKKOS_LAMBDA (const int receiver, int& update) {
            update += active_num_neigh(receiver);
        },
        active_edges);
    const auto invalid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), all_interactions_geometry_invalid);
    if (invalid_host(0) != 0)
        throw std::invalid_argument(
            "Streamed all positions produced an invalid edge geometry.");

    prepare_streamed_edge_schedule(num_nodes, active_edges, active_num_neigh);
    const auto candidate_first_neigh = all_interactions_candidate_first_neigh;
    const auto candidate_num_neigh = all_interactions_candidate_num_neigh;
    const auto candidate_sources = all_interactions_candidate_neigh_indices;
    const auto candidate_types = all_interactions_candidate_neigh_types;
    const auto active_first_neigh = streamed_first_neigh;
    auto active_sources = all_interactions_active_neigh_indices;
    auto active_types = all_interactions_active_neigh_types;
    auto active_receivers = all_interactions_active_edge_receivers;
    auto active_xyz = all_interactions_active_xyz;
    auto active_r = all_interactions_active_r;
    Kokkos::parallel_for(
        "MACEKokkos::all_interactions_compact_active_edges",
        num_nodes,
        KOKKOS_LAMBDA (const int receiver) {
            const int candidate_begin = candidate_first_neigh(receiver);
            const int candidate_end =
                candidate_begin+candidate_num_neigh(receiver);
            int output = active_first_neigh(receiver);
            for (int edge=candidate_begin; edge<candidate_end; ++edge) {
                if (candidate_active(edge) == 0)
                    continue;
                active_receivers(output) = receiver;
                active_sources(output) = candidate_sources(edge);
                active_types(output) = candidate_types(edge);
                active_r(output) = candidate_r(edge);
                for (int component=0; component<3; ++component)
                    active_xyz(3*output+component) =
                        candidate_xyz(3*edge+component);
                output += 1;
            }
        });

    const auto node_types = Kokkos::View<const int*>(
        all_interactions_candidate_node_types);
    const auto num_neigh = Kokkos::View<const int*>(active_num_neigh);
    const auto neigh_indices = Kokkos::subview(
        all_interactions_active_neigh_indices,
        Kokkos::make_pair(std::size_t(0), static_cast<std::size_t>(active_edges)));
    const auto neigh_types = Kokkos::subview(
        all_interactions_active_neigh_types,
        Kokkos::make_pair(std::size_t(0), static_cast<std::size_t>(active_edges)));
    const auto xyz = Kokkos::subview(
        all_interactions_active_xyz,
        Kokkos::make_pair(
            std::size_t(0), 3*static_cast<std::size_t>(active_edges)));
    const auto r = Kokkos::subview(
        all_interactions_active_r,
        Kokkos::make_pair(std::size_t(0), static_cast<std::size_t>(active_edges)));
    compute_node_energies_forces(
        num_nodes, node_types, num_neigh, neigh_indices, neigh_types, xyz, r,
        0, true);
    all_interactions_active_edge_count = active_edges;
    all_interactions_prepared_evaluation_count += 1;
    all_interactions_geometry_update_count += 1;
    all_interactions_completed_graph_generation = graph_generation;
}

template <typename Precision>
void MACEKokkos<Precision>::invalidate_factorized_prepared_graph()
{
    factorized_prepared_graph_generation = 0;
    factorized_prepared_geometry_graph_generation = 0;
    factorized_prepared_geometry_fractional = false;
    factorized_prepared_geometry_uses_shifts = false;
    invalidate_factorized_operator_benchmark();
    factorized_completed_evaluation_epoch = 0;
    factorized_completed_evaluation_graph_generation = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::release_dual_layer_source_schedule()
{
    const std::size_t released_bytes = dual_layer_source_schedule_bytes;
    const bool allocated = dual_layer_tile_segment_offsets.extent(0) != 0
        || dual_layer_segment_source_ids.extent(0) != 0
        || dual_layer_segment_edge_offsets.extent(0) != 0
        || dual_layer_source_edges.extent(0) != 0;
    if (allocated) {
        factorized_execution_space.fence(
            "Release previous dual-layer source schedule");
        factorized_preparation_fence_count += 1;
    }
    if (factorized_schedule_bytes >= released_bytes)
        factorized_schedule_bytes -= released_bytes;
    else
        factorized_schedule_bytes = 0;
    if (low_memory_selected_estimate >= released_bytes)
        low_memory_selected_estimate -= released_bytes;
    else
        low_memory_selected_estimate = 0;
    if (execution_planned_bytes >= released_bytes)
        execution_planned_bytes -= released_bytes;
    else
        execution_planned_bytes = 0;
    dual_layer_tile_segment_offsets = {};
    dual_layer_segment_source_ids = {};
    dual_layer_segment_edge_offsets = {};
    dual_layer_source_edges = {};
    dual_layer_tile_segment_offsets_host.clear();
    dual_layer_source_segments = 0;
    dual_layer_active_tile = 0;
    dual_layer_active_segment_begin = 0;
    dual_layer_active_segment_count = 0;
    dual_layer_source_schedule_bytes = 0;
    dual_layer_source_schedule_preparation_explicit_scratch_bytes = 0;
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph(
    const int num_nodes,
    const std::span<const int> node_types,
    const std::span<const int> num_neigh,
    const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    return prepare_factorized_graph(
        num_nodes, num_nodes, node_types, num_neigh, neigh_indices, neigh_types);
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph(
    const int num_receivers,
    const int num_feature_nodes,
    const std::span<const int> node_types,
    const std::span<const int> num_neigh,
    const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers
        || node_types.size() != static_cast<std::size_t>(num_receivers)
        || num_neigh.size() != static_cast<std::size_t>(num_receivers)
        || neigh_indices.size() != neigh_types.size())
        throw std::invalid_argument(
            "Prepared Execution R1 graph extents are inconsistent.");
    std::vector<int> receiver_feature_indices(
        static_cast<std::size_t>(num_receivers));
    std::iota(
        receiver_feature_indices.begin(), receiver_feature_indices.end(), 0);
    std::vector<int> feature_types(
        static_cast<std::size_t>(num_feature_nodes), 0);
    std::copy(node_types.begin(), node_types.end(), feature_types.begin());
    for (std::size_t edge=0; edge<neigh_indices.size(); ++edge) {
        const int source = neigh_indices[edge];
        if (source >= 0 && source < num_feature_nodes)
            feature_types[static_cast<std::size_t>(source)] = neigh_types[edge];
    }
    return prepare_factorized_graph(
        num_receivers, num_feature_nodes, receiver_feature_indices, node_types,
        feature_types, num_neigh, neigh_indices, neigh_types);
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph(
    const int num_receivers,
    const int num_feature_nodes,
    const std::span<const int> receiver_feature_indices,
    const std::span<const int> node_types,
    const std::span<const int> feature_types,
    const std::span<const int> num_neigh,
    const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types)
{
    const auto start = std::chrono::steady_clock::now();
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        throw std::logic_error(
            "A prepared factorized graph cannot change during a distributed evaluation.");
    if (!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "Prepared Execution R1 graphs require streamed_edges='factorized'.");
    if (num_receivers < 0 || num_feature_nodes < num_receivers
        || receiver_feature_indices.size()
            != static_cast<std::size_t>(num_receivers)
        || node_types.size() != static_cast<std::size_t>(num_receivers)
        || feature_types.size() != static_cast<std::size_t>(num_feature_nodes)
        || num_neigh.size() != static_cast<std::size_t>(num_receivers)
        || neigh_indices.size() != neigh_types.size())
        throw std::invalid_argument(
            "Prepared Execution R1 graph extents are inconsistent.");
    validate_graph_cardinality(
        static_cast<std::size_t>(num_receivers),
        static_cast<std::size_t>(num_feature_nodes),
        neigh_indices.size());

    const auto prepared_receiver_feature_indices = std::vector<int>(
        receiver_feature_indices.begin(), receiver_feature_indices.end());
    const auto prepared_node_types = std::vector<int>(
        node_types.begin(), node_types.end());
    const auto prepared_feature_types = std::vector<int>(
        feature_types.begin(), feature_types.end());
    const auto prepared_num_neigh = std::vector<int>(
        num_neigh.begin(), num_neigh.end());
    const auto prepared_neigh_indices = std::vector<int>(
        neigh_indices.begin(), neigh_indices.end());
    const auto prepared_neigh_types = std::vector<int>(
        neigh_types.begin(), neigh_types.end());
    for (const int type : prepared_node_types)
        if (type < 0 || type >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Prepared Execution R1 node type is out of range.");
    for (const int type : prepared_feature_types)
        if (type < 0 || type >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Prepared Execution R1 feature type is out of range.");
    for (std::size_t receiver=0;
         receiver<prepared_receiver_feature_indices.size(); ++receiver) {
        const int feature = prepared_receiver_feature_indices[receiver];
        if (feature < 0 || feature >= num_feature_nodes)
            throw std::out_of_range(
                "Prepared Execution R1 receiver feature index is out of range.");
        if (prepared_node_types[receiver]
            != prepared_feature_types[static_cast<std::size_t>(feature)])
            throw std::invalid_argument(
                "Prepared Execution R1 receiver and feature types disagree.");
    }
    {
        std::vector<unsigned char> receiver_features_seen(
            static_cast<std::size_t>(num_feature_nodes), 0);
        for (const int feature : prepared_receiver_feature_indices) {
            if (receiver_features_seen[static_cast<std::size_t>(feature)] != 0)
                throw std::invalid_argument(
                    "Prepared Execution R1 receiver feature indices must be unique.");
            receiver_features_seen[static_cast<std::size_t>(feature)] = 1;
        }
    }
    for (const int type : prepared_neigh_types)
        if (type < 0 || type >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Prepared Execution R1 neighbor type is out of range.");
    for (const int source : prepared_neigh_indices)
        if (source < 0 || source >= num_feature_nodes)
            throw std::out_of_range(
                "Prepared Execution R1 source index is out of range.");

    const bool same_identity = factorized_prepared_graph_generation != 0
        && num_feature_nodes == execution_prepared_num_feature_nodes
        && prepared_receiver_feature_indices
            == execution_prepared_receiver_feature_indices_host
        && prepared_node_types == execution_prepared_node_types_host
        && prepared_feature_types == execution_prepared_feature_types_host
        && prepared_num_neigh == execution_prepared_num_neigh_host
        && prepared_neigh_indices == execution_prepared_neigh_indices_host
        && prepared_neigh_types == execution_prepared_neigh_types_host
        && !factorized_schedule_dirty;
    if (same_identity) {
        factorized_last_graph_prepare_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-start).count();
        return factorized_prepared_graph_generation;
    }

    invalidate_factorized_prepared_graph();
    release_dual_layer_source_schedule();
    select_low_memory_policy_for_graph(
        num_receivers, num_feature_nodes, prepared_neigh_indices.size(),
        prepared_num_neigh);
    if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
        for (std::size_t edge=0; edge<prepared_neigh_indices.size(); ++edge)
            if (prepared_neigh_types[edge]
                != prepared_feature_types[prepared_neigh_indices[edge]])
                throw std::invalid_argument(
                    "Fixed-workspace tiled neighbor types must match source feature types.");
    }
    const std::size_t graph_replacements_before =
        factorized_graph_device_replacement_count;
    prepare_factorized_schedule_host(
        num_receivers, num_feature_nodes,
        prepared_num_neigh, prepared_neigh_indices);
    if (dual_layer_tiled_plan_active)
        prepare_dual_layer_tile_source_schedule_host(
            num_receivers, num_feature_nodes,
            prepared_num_neigh, prepared_neigh_indices);
    if (single_layer_tiled_plan_active)
        prepare_single_layer_tiled_workspace(num_receivers);
    if (dual_layer_tiled_plan_active)
        prepare_dual_layer_tiled_workspace(num_receivers, num_feature_nodes);
    reserve_execution_geometry_workspace(
        static_cast<int>(prepared_neigh_indices.size()));
    copy_int_prefix(
        factorized_execution_space,
        execution_prepared_receiver_feature_indices_storage,
        execution_prepared_receiver_feature_indices,
        prepared_receiver_feature_indices);
    copy_int_prefix(
        factorized_execution_space, execution_prepared_node_types_storage,
        execution_prepared_node_types, prepared_node_types);
    copy_int_prefix(
        factorized_execution_space, execution_prepared_feature_types_storage,
        execution_prepared_feature_types, prepared_feature_types);
    copy_int_prefix(
        factorized_execution_space, execution_prepared_num_neigh_storage,
        execution_prepared_num_neigh, prepared_num_neigh);
    copy_int_prefix(
        factorized_execution_space, execution_prepared_neigh_indices_storage,
        execution_prepared_neigh_indices, prepared_neigh_indices);
    if (!single_layer_tiled_plan_active && !dual_layer_tiled_plan_active)
        copy_int_prefix(
            factorized_execution_space, execution_prepared_neigh_types_storage,
            execution_prepared_neigh_types, prepared_neigh_types);
    else
        execution_prepared_neigh_types = {};
    factorized_execution_space.fence("Execution R1 explicit graph preparation");
    factorized_preparation_fence_count += 1;
    if (factorized_graph_device_replacement_count
            == graph_replacements_before)
        factorized_graph_device_update_count += 1;

    execution_prepared_receiver_feature_indices_host =
        prepared_receiver_feature_indices;
    execution_prepared_node_types_host = prepared_node_types;
    execution_prepared_feature_types_host = prepared_feature_types;
    execution_prepared_num_neigh_host = prepared_num_neigh;
    execution_prepared_neigh_indices_host = prepared_neigh_indices;
    execution_prepared_neigh_types_host = prepared_neigh_types;
    execution_prepared_num_feature_nodes = num_feature_nodes;
    factorized_graph_generation_counter += 1;
    if (factorized_graph_generation_counter == 0)
        factorized_graph_generation_counter += 1;
    factorized_prepared_graph_generation = factorized_graph_generation_counter;
    factorized_prepared_graph_count += 1;
    factorized_last_graph_prepare_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    return factorized_prepared_graph_generation;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_single_layer_tiled_workspace(
    const int num_receivers)
{
    if (!single_layer_tiled_plan_active)
        return;
    const int capacity = std::min(
        num_receivers, single_layer_workspace_receiver_limit);
    if (capacity <= 0)
        throw std::invalid_argument(
            "Single-layer tiled execution requires at least one receiver.");
    int edge_capacity = 0;
    for (int receiver_begin=0; receiver_begin<num_receivers;) {
        const int receiver_end = receiver_begin
            +std::min(capacity, num_receivers-receiver_begin);
        const int edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int edge_end = receiver_end == num_receivers
            ? execution_active_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        edge_capacity = std::max(edge_capacity, edge_end-edge_begin);
        receiver_begin = receiver_end;
    }
    single_layer_workspace_active_edge_capacity = edge_capacity;

    const std::size_t precision_per_receiver =
        (static_cast<std::size_t>(num_lm)
         +2*static_cast<std::size_t>(num_LM)+1)
        *static_cast<std::size_t>(num_channels);
    single_layer_workspace_receiver_bytes = checked_workspace_sum(
        "single-layer workspace receiver bytes",
        {precision_per_receiver*sizeof(Precision),
         static_cast<std::size_t>(num_channels)*sizeof(double),
         sizeof(double), sizeof(int)});
    const std::size_t precision_bytes = checked_workspace_product(
        "single-layer precision workspace bytes",
        {static_cast<std::size_t>(capacity), precision_per_receiver,
         sizeof(Precision)});
    const std::size_t double_offset = checked_workspace_align_up(
        "single-layer FP64 workspace alignment", precision_bytes,
        alignof(double));
    const std::size_t required_bytes = checked_workspace_sum(
        "single-layer workspace bytes",
        {double_offset,
         checked_workspace_product(
             "single-layer H2 slot",
             {static_cast<std::size_t>(capacity),
              static_cast<std::size_t>(num_channels), sizeof(double)}),
         checked_workspace_product(
             "single-layer density slot",
             {static_cast<std::size_t>(capacity), sizeof(double)}),
         checked_workspace_product(
             "single-layer receiver-offset slot",
             {static_cast<std::size_t>(capacity), sizeof(int)})});
    single_layer_workspace_active_capacity = capacity;
    single_layer_workspace_planned_edge_capacity = std::max(
        single_layer_workspace_planned_edge_capacity, edge_capacity);
    const std::size_t geometry_bytes_per_edge =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64
        ? std::size_t(3)*sizeof(Precision)
        : std::size_t(3)*sizeof(double);
    single_layer_workspace_edge_bytes = checked_workspace_sum(
        "single-layer edge workspace bytes",
        {checked_workspace_product(
             "single-layer harmonic workspace bytes",
             {static_cast<std::size_t>(
                  single_layer_workspace_planned_edge_capacity),
              static_cast<std::size_t>(num_lm), sizeof(Precision)}),
         checked_workspace_product(
             "single-layer directed-force workspace bytes",
             {static_cast<std::size_t>(
                  single_layer_workspace_planned_edge_capacity),
              std::size_t(3), sizeof(double)}),
         checked_workspace_product(
             "single-layer geometry workspace bytes",
             {static_cast<std::size_t>(
                  single_layer_workspace_planned_edge_capacity),
              geometry_bytes_per_edge}),
         checked_workspace_product(
             "single-layer radius workspace bytes",
             {static_cast<std::size_t>(
                  single_layer_workspace_planned_edge_capacity),
              sizeof(double)}),
         checked_workspace_product(
             "single-layer neighbor-type workspace bytes",
             {static_cast<std::size_t>(
                  single_layer_workspace_planned_edge_capacity),
              sizeof(int)})});
    if (single_layer_workspace_arena.data() != nullptr
        && single_layer_workspace_planned_capacity >= capacity) {
        single_layer_workspace_reuses += 1;
        return;
    }

    factorized_execution_space.fence("Replace single-layer tiled workspace");
    A0 = {};
    A0_adj = {};
    M0 = {};
    M0_adj = {};
    H1 = {};
    H1_adj = {};
    M1 = {};
    M1_adj = {};
    H2 = {};
    H2_adj = {};
    standard_r0_density_state = {};
    mh0_a0_scale_adjoint = {};
    single_layer_workspace_a0 = {};
    single_layer_workspace_equivariant_a = {};
    single_layer_workspace_equivariant_b = {};
    single_layer_workspace_m1 = {};
    single_layer_workspace_h2 = {};
    single_layer_workspace_density = {};
    single_layer_workspace_first_neigh = {};
    single_layer_workspace_arena = {};

    const std::size_t arena_words = (required_bytes+sizeof(std::uint64_t)-1)
        /sizeof(std::uint64_t);
    single_layer_workspace_arena = decltype(single_layer_workspace_arena)(
        Kokkos::view_alloc(
            "Single-layer tiled workspace", Kokkos::WithoutInitializing),
        arena_words);
    auto* raw = reinterpret_cast<unsigned char*>(
        single_layer_workspace_arena.data());
    std::size_t offset = 0;
    single_layer_workspace_a0 = make_tracked_workspace_alias<
        decltype(single_layer_workspace_a0)>(
            single_layer_workspace_arena,
            reinterpret_cast<Precision*>(raw+offset),
            capacity, num_lm, num_channels);
    offset += checked_workspace_product(
        "single-layer A0 slot",
        {static_cast<std::size_t>(capacity), static_cast<std::size_t>(num_lm),
         static_cast<std::size_t>(num_channels), sizeof(Precision)});
    single_layer_workspace_equivariant_a = make_tracked_workspace_alias<
        decltype(single_layer_workspace_equivariant_a)>(
            single_layer_workspace_arena,
            reinterpret_cast<Precision*>(raw+offset),
            capacity, num_LM, num_channels);
    offset += checked_workspace_product(
        "single-layer equivariant A slot",
        {static_cast<std::size_t>(capacity), static_cast<std::size_t>(num_LM),
         static_cast<std::size_t>(num_channels), sizeof(Precision)});
    single_layer_workspace_equivariant_b = make_tracked_workspace_alias<
        decltype(single_layer_workspace_equivariant_b)>(
            single_layer_workspace_arena,
            reinterpret_cast<Precision*>(raw+offset),
            capacity, num_LM, num_channels);
    offset += checked_workspace_product(
        "single-layer equivariant B slot",
        {static_cast<std::size_t>(capacity), static_cast<std::size_t>(num_LM),
         static_cast<std::size_t>(num_channels), sizeof(Precision)});
    single_layer_workspace_m1 = make_tracked_workspace_alias<
        decltype(single_layer_workspace_m1)>(
        single_layer_workspace_arena,
        reinterpret_cast<Precision*>(raw+offset), capacity, num_channels);
    offset += checked_workspace_product(
        "single-layer M1 slot",
        {static_cast<std::size_t>(capacity),
         static_cast<std::size_t>(num_channels), sizeof(Precision)});
    offset = checked_workspace_align_up(
        "single-layer FP64 workspace alignment", offset, alignof(double));
    single_layer_workspace_h2 = make_tracked_workspace_alias<
        decltype(single_layer_workspace_h2)>(
        single_layer_workspace_arena,
        reinterpret_cast<double*>(raw+offset), capacity, num_channels);
    offset += checked_workspace_product(
        "single-layer H2 slot",
        {static_cast<std::size_t>(capacity),
         static_cast<std::size_t>(num_channels), sizeof(double)});
    single_layer_workspace_density = make_tracked_workspace_alias<
        decltype(single_layer_workspace_density)>(
        single_layer_workspace_arena,
        reinterpret_cast<double*>(raw+offset), capacity);
    offset += checked_workspace_product(
        "single-layer density slot",
        {static_cast<std::size_t>(capacity), sizeof(double)});
    single_layer_workspace_first_neigh = make_tracked_workspace_alias<
        decltype(single_layer_workspace_first_neigh)>(
        single_layer_workspace_arena,
        reinterpret_cast<int*>(raw+offset), capacity);
    offset += checked_workspace_product(
        "single-layer receiver-offset slot",
        {static_cast<std::size_t>(capacity), sizeof(int)});
    if (offset != required_bytes)
        throw std::logic_error("Single-layer workspace slot accounting mismatch.");

    single_layer_workspace_planned_capacity = capacity;
    single_layer_workspace_payload_bytes = required_bytes;
    single_layer_workspace_replacements += 1;
    bind_single_layer_tiled_workspace_batch(capacity);
}

template <typename Precision>
void MACEKokkos<Precision>::bind_single_layer_tiled_workspace_batch(
    const int active_receivers)
{
    if (active_receivers <= 0
        || active_receivers > single_layer_workspace_planned_capacity)
        throw std::logic_error(
            "Single-layer workspace batch exceeds the prepared capacity.");
    const auto rows = Kokkos::make_pair(
        std::size_t(0), static_cast<std::size_t>(active_receivers));
    A0 = Kokkos::subview(
        single_layer_workspace_a0, rows, Kokkos::ALL, Kokkos::ALL);
    M0 = Kokkos::subview(
        single_layer_workspace_equivariant_a,
        rows, Kokkos::ALL, Kokkos::ALL);
    H1 = Kokkos::subview(
        single_layer_workspace_equivariant_b,
        rows, Kokkos::ALL, Kokkos::ALL);
    M1 = Kokkos::subview(single_layer_workspace_m1, rows, Kokkos::ALL);
    H2 = Kokkos::subview(single_layer_workspace_h2, rows, Kokkos::ALL);
    standard_r0_density_state = Kokkos::subview(
        single_layer_workspace_density, rows);
    streamed_first_neigh = Kokkos::subview(
        single_layer_workspace_first_neigh, rows);

    A0_adj = {};
    M0_adj = {};
    H1_adj = {};
    M1_adj = {};
    H2_adj = {};
    // H2 is dead before M0 reverse. Reuse its leading doubles for the density
    // scale adjoint while retaining the density value in its dedicated slot.
    mh0_a0_scale_adjoint = make_tracked_workspace_alias<
        decltype(mh0_a0_scale_adjoint)>(
        single_layer_workspace_arena,
        single_layer_workspace_h2.data(), active_receivers);
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_dual_layer_tiled_workspace(
    const int num_receivers, const int num_feature_nodes)
{
    if (!dual_layer_tiled_plan_active)
        return;
    const int capacity = std::min(
        num_receivers, dual_layer_workspace_receiver_limit);
    if (capacity <= 0)
        throw std::invalid_argument(
            "Dual-layer tiled execution requires at least one receiver.");
    int edge_capacity = 0;
    for (int receiver_begin=0; receiver_begin<num_receivers;) {
        const int receiver_end = receiver_begin
            +std::min(capacity, num_receivers-receiver_begin);
        const int edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int edge_end = receiver_end == num_receivers
            ? execution_active_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        edge_capacity = std::max(edge_capacity, edge_end-edge_begin);
        receiver_begin = receiver_end;
    }
    dual_layer_workspace_active_capacity = capacity;
    dual_layer_workspace_active_edge_capacity = edge_capacity;
    dual_layer_workspace_planned_edge_capacity = std::max(
        dual_layer_workspace_planned_edge_capacity, edge_capacity);

    const std::size_t channels = static_cast<std::size_t>(num_channels);
    const std::size_t receivers = static_cast<std::size_t>(capacity);
    const std::size_t a_slot_elements = checked_workspace_product(
        "dual-layer A slot", {receivers, static_cast<std::size_t>(num_lm), channels});
    const std::size_t equivariant_elements = checked_workspace_product(
        "dual-layer equivariant slot",
        {receivers, static_cast<std::size_t>(num_LM), channels});
    const std::size_t phi_elements = checked_workspace_product(
        "dual-layer Phi1 slot",
        {receivers, static_cast<std::size_t>(num_lme),
         static_cast<std::size_t>(phi1_channel_tile_size)});
    const std::size_t m1_elements = checked_workspace_product(
        "dual-layer M1 slot", {receivers, channels});
    const std::size_t precision_bytes = checked_workspace_product(
        "dual-layer precision slots",
        {checked_workspace_sum(
             "dual-layer precision slots",
             {a_slot_elements, std::size_t(2)*equivariant_elements,
              phi_elements, m1_elements}),
         sizeof(Precision)});
    const std::size_t double_offset = checked_workspace_align_up(
        "dual-layer FP64 workspace alignment", precision_bytes, alignof(double));
    const std::size_t h2_bytes = checked_workspace_product(
        "dual-layer H2 slot", {receivers, channels, sizeof(double)});
    const std::size_t density_bytes = checked_workspace_product(
        "dual-layer density slots", {receivers, std::size_t(2), sizeof(double)});
    const std::size_t first_neigh_bytes = checked_workspace_product(
        "dual-layer receiver offsets", {receivers, sizeof(int)});
    const std::size_t required_bytes = checked_workspace_sum(
        "dual-layer workspace",
        {double_offset, h2_bytes, density_bytes, first_neigh_bytes});
    dual_layer_workspace_receiver_bytes = required_bytes/receivers;
    const std::size_t geometry_bytes_per_edge =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64
        ? std::size_t(3)*sizeof(Precision)
        : std::size_t(3)*sizeof(double);
    dual_layer_workspace_edge_bytes = checked_workspace_product(
        "dual-layer edge workspace",
        {static_cast<std::size_t>(dual_layer_workspace_planned_edge_capacity),
         checked_workspace_sum(
             "dual-layer edge workspace per edge",
             {static_cast<std::size_t>(num_lm)*sizeof(Precision),
              std::size_t(3)*sizeof(double), geometry_bytes_per_edge,
              sizeof(double), std::size_t(2)*sizeof(int)})});

    const std::size_t graph_elements = checked_workspace_product(
        "dual-layer graph H1 state",
        {static_cast<std::size_t>(num_feature_nodes),
         static_cast<std::size_t>(num_LM), channels});
    const bool graph_state_fits =
        dual_layer_graph_h1.size() >= graph_elements
        && dual_layer_graph_h1_adjoint.size() >= graph_elements;
    const bool arena_fits = dual_layer_workspace_arena.data() != nullptr
        && dual_layer_workspace_planned_capacity >= capacity;
    const bool edge_receiver_fits = dual_layer_edge_local_receivers.extent(0)
        >= static_cast<std::size_t>(dual_layer_workspace_planned_edge_capacity);
    if (arena_fits && graph_state_fits && edge_receiver_fits) {
        dual_layer_workspace_reuses += 1;
        bind_dual_layer_phase1_workspace(0, capacity);
        return;
    }

    factorized_execution_space.fence("Replace dual-layer tiled workspace");
    A0 = {};
    A0_adj = {};
    M0 = {};
    M0_adj = {};
    H1 = {};
    H1_adj = {};
    Phi1 = {};
    dPhi1 = {};
    A1 = {};
    A1_adj = {};
    M1 = {};
    M1_adj = {};
    H2 = {};
    H2_adj = {};
    standard_r0_density_state = {};
    mh0_a0_scale_adjoint = {};
    mh0_a1_scale_adjoint = {};

    if (!arena_fits) {
        dual_layer_workspace_a0 = {};
        dual_layer_workspace_equivariant_a = {};
        dual_layer_workspace_equivariant_b = {};
        dual_layer_workspace_a1 = {};
        dual_layer_workspace_phi1 = {};
        dual_layer_workspace_m1 = {};
        dual_layer_workspace_h2 = {};
        dual_layer_workspace_density_a0 = {};
        dual_layer_workspace_density_a1 = {};
        dual_layer_workspace_first_neigh = {};
        dual_layer_workspace_arena = {};
        const std::size_t words = (required_bytes+sizeof(std::uint64_t)-1)
            /sizeof(std::uint64_t);
        dual_layer_workspace_arena = decltype(dual_layer_workspace_arena)(
            Kokkos::view_alloc(
                "Dual-layer tiled workspace", Kokkos::WithoutInitializing),
            words);
        auto* raw = reinterpret_cast<unsigned char*>(
            dual_layer_workspace_arena.data());
        std::size_t offset = 0;
        auto* a_slot = reinterpret_cast<Precision*>(raw+offset);
        dual_layer_workspace_a0 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_a0)>(
                dual_layer_workspace_arena, a_slot,
                capacity, num_lm, num_channels);
        dual_layer_workspace_a1 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_a1)>(
                dual_layer_workspace_arena, a_slot,
                capacity, num_lm, num_channels);
        offset += a_slot_elements*sizeof(Precision);
        dual_layer_workspace_equivariant_a = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_equivariant_a)>(
                dual_layer_workspace_arena,
                reinterpret_cast<Precision*>(raw+offset),
                capacity, num_LM, num_channels);
        offset += equivariant_elements*sizeof(Precision);
        dual_layer_workspace_equivariant_b = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_equivariant_b)>(
                dual_layer_workspace_arena,
                reinterpret_cast<Precision*>(raw+offset),
                capacity, num_LM, num_channels);
        offset += equivariant_elements*sizeof(Precision);
        dual_layer_workspace_phi1 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_phi1)>(
                dual_layer_workspace_arena,
                reinterpret_cast<Precision*>(raw+offset),
                capacity, num_lme, phi1_channel_tile_size);
        offset += phi_elements*sizeof(Precision);
        dual_layer_workspace_m1 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_m1)>(
                dual_layer_workspace_arena,
                reinterpret_cast<Precision*>(raw+offset),
                capacity, num_channels);
        offset += m1_elements*sizeof(Precision);
        offset = checked_workspace_align_up(
            "dual-layer FP64 workspace alignment", offset, alignof(double));
        dual_layer_workspace_h2 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_h2)>(
                dual_layer_workspace_arena,
                reinterpret_cast<double*>(raw+offset), capacity, num_channels);
        offset += h2_bytes;
        dual_layer_workspace_density_a0 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_density_a0)>(
                dual_layer_workspace_arena,
                reinterpret_cast<double*>(raw+offset), capacity);
        offset += receivers*sizeof(double);
        dual_layer_workspace_density_a1 = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_density_a1)>(
                dual_layer_workspace_arena,
                reinterpret_cast<double*>(raw+offset), capacity);
        offset += receivers*sizeof(double);
        dual_layer_workspace_first_neigh = make_tracked_workspace_alias<
            decltype(dual_layer_workspace_first_neigh)>(
                dual_layer_workspace_arena,
                reinterpret_cast<int*>(raw+offset), capacity);
        offset += first_neigh_bytes;
        if (offset != required_bytes)
            throw std::logic_error(
                "Dual-layer workspace slot accounting mismatch.");
        dual_layer_workspace_planned_capacity = capacity;
        dual_layer_workspace_payload_bytes = required_bytes;
    }
    if (!graph_state_fits) {
        dual_layer_graph_h1 = decltype(dual_layer_graph_h1)(
            Kokkos::view_alloc(
                "Dual-layer graph H1", Kokkos::WithoutInitializing),
            num_feature_nodes, num_LM, num_channels);
        dual_layer_graph_h1_adjoint = decltype(dual_layer_graph_h1_adjoint)(
            Kokkos::view_alloc(
                "Dual-layer graph H1 adjoint", Kokkos::WithoutInitializing),
            num_feature_nodes, num_LM, num_channels);
    }
    if (!edge_receiver_fits) {
        dual_layer_edge_local_receivers = Kokkos::View<int*>(
            Kokkos::view_alloc(
                "Dual-layer tile-local edge receivers",
                Kokkos::WithoutInitializing),
            static_cast<std::size_t>(dual_layer_workspace_planned_edge_capacity));
    }
    dual_layer_workspace_replacements += 1;
    bind_dual_layer_phase1_workspace(0, capacity);
}

template <typename Precision>
void MACEKokkos<Precision>::bind_dual_layer_phase1_workspace(
    const int receiver_begin, const int receiver_count)
{
    if (receiver_count <= 0
        || receiver_count > dual_layer_workspace_planned_capacity)
        throw std::logic_error(
            "Dual-layer phase-1 batch exceeds the prepared capacity.");
    const auto local_rows = Kokkos::make_pair(
        std::size_t(0), static_cast<std::size_t>(receiver_count));
    A0 = Kokkos::subview(
        dual_layer_workspace_a0, local_rows, Kokkos::ALL, Kokkos::ALL);
    M0 = Kokkos::subview(
        dual_layer_workspace_equivariant_a,
        local_rows, Kokkos::ALL, Kokkos::ALL);
    H1 = Kokkos::subview(
        dual_layer_workspace_equivariant_b,
        local_rows, Kokkos::ALL, Kokkos::ALL);
    H1_adj = M0;
    standard_r0_density_state = Kokkos::subview(
        dual_layer_workspace_density_a0, local_rows);
    streamed_first_neigh = Kokkos::subview(
        dual_layer_workspace_first_neigh, local_rows);
    A0_adj = {};
    M0_adj = {};
}

template <typename Precision>
void MACEKokkos<Precision>::bind_dual_layer_phase2_workspace(
    const int receiver_begin, const int receiver_count)
{
    if (receiver_count <= 0
        || receiver_count > dual_layer_workspace_planned_capacity)
        throw std::logic_error(
            "Dual-layer phase-2 batch exceeds the prepared capacity.");
    const auto local_rows = Kokkos::make_pair(
        std::size_t(0), static_cast<std::size_t>(receiver_count));
    A1 = Kokkos::subview(
        dual_layer_workspace_a1, local_rows, Kokkos::ALL, Kokkos::ALL);
    Phi1 = Kokkos::subview(
        dual_layer_workspace_phi1, local_rows, Kokkos::ALL, Kokkos::ALL);
    M1 = Kokkos::subview(
        dual_layer_workspace_m1, local_rows, Kokkos::ALL);
    H2 = Kokkos::subview(
        dual_layer_workspace_h2, local_rows, Kokkos::ALL);
    H1 = Kokkos::subview(
        dual_layer_workspace_equivariant_a,
        local_rows, Kokkos::ALL, Kokkos::ALL);
    H1_adj = Kokkos::subview(
        dual_layer_workspace_equivariant_b,
        local_rows, Kokkos::ALL, Kokkos::ALL);
    M0 = {};
    streamed_first_neigh = Kokkos::subview(
        dual_layer_workspace_first_neigh, local_rows);
    A1_adj = {};
    dPhi1 = {};
    M1_adj = {};
    H2_adj = {};
    mh0_a1_scale_adjoint = make_tracked_workspace_alias<
        decltype(mh0_a1_scale_adjoint)>(
            dual_layer_workspace_arena,
            dual_layer_workspace_h2.data(), receiver_count);
    (void)receiver_begin;
}

template <typename Precision>
void MACEKokkos<Precision>::bind_dual_layer_phase3_workspace(
    const int receiver_begin, const int receiver_count)
{
    bind_dual_layer_phase1_workspace(receiver_begin, receiver_count);
    H1_adj = H1;
    mh0_a0_scale_adjoint = make_tracked_workspace_alias<
        decltype(mh0_a0_scale_adjoint)>(
            dual_layer_workspace_arena,
            dual_layer_workspace_h2.data(), receiver_count);
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_single_layer_tiled_geometry(
    const int receiver_begin,
    const int receiver_count,
    const int edge_begin,
    const int edge_count)
{
    if (!single_layer_tiled_plan_active && !dual_layer_tiled_plan_active)
        throw std::logic_error(
            "Single-layer tiled geometry requires tiled execution.");
    if (receiver_begin < 0 || receiver_count <= 0
        || edge_begin < 0 || edge_count < 0)
        throw std::invalid_argument(
            "Single-layer tiled geometry has invalid edge bounds.");
    const std::size_t receiver_end = static_cast<std::size_t>(receiver_begin)
        +static_cast<std::size_t>(receiver_count);
    const bool compact_geometry = use_compact_edge_geometry();
    const bool explicit_host_geometry =
        single_layer_explicit_xyz_host.extent(0) != 0
        || single_layer_explicit_r_host.extent(0) != 0;
    const bool explicit_device_geometry =
        single_layer_explicit_xyz_device.extent(0) != 0
        || single_layer_explicit_r_device.extent(0) != 0;
    const bool explicit_edge_geometry =
        explicit_host_geometry || explicit_device_geometry;
    const bool explicit_feature_positions =
        single_layer_explicit_feature_positions_device.extent(0) != 0;
    const std::size_t edge_end = static_cast<std::size_t>(edge_begin)
        +static_cast<std::size_t>(edge_count);
    const bool explicit_geometry_fits =
        (explicit_host_geometry
            ? single_layer_explicit_xyz_host.extent(0) >= 3*edge_end
                && single_layer_explicit_r_host.extent(0) >= edge_end
            : single_layer_explicit_xyz_device.extent(0) >= 3*edge_end
                && single_layer_explicit_r_device.extent(0) >= edge_end)
        && single_layer_workspace_xyz.extent(0)
            >= static_cast<std::size_t>(3)*edge_count;
    const bool explicit_feature_positions_fit =
        single_layer_explicit_feature_positions_device.extent(0)
            >= std::size_t(3)*static_cast<std::size_t>(
                execution_prepared_num_feature_nodes)
        && execution_prepared_receiver_feature_indices.extent(0)
            >= receiver_end;
    const bool retained_geometry_fits = factorized_prepared_geometry_uses_shifts
        ? execution_prepared_edge_shifts.extent(0) >= 3*edge_end
            && execution_prepared_reference_positions.extent(0)
                >= std::size_t(3)*execution_prepared_node_types.extent(0)
        : execution_prepared_reference_xyz.extent(0) >= 3*edge_end;
    const auto feature_types = execution_prepared_feature_types.extent(0) == 0
        ? execution_prepared_node_types : execution_prepared_feature_types;
    if (receiver_end > execution_prepared_node_types.extent(0)
        || edge_end > execution_prepared_neigh_indices.extent(0)
        || feature_types.extent(0)
            < static_cast<std::size_t>(execution_prepared_num_feature_nodes)
        || (explicit_edge_geometry
            ? !explicit_geometry_fits
            : explicit_feature_positions
                ? !explicit_feature_positions_fit
                : !retained_geometry_fits)
        || (compact_geometry
            ? execution_prepared_unit_direction.extent(0)
            : single_layer_workspace_xyz.extent(0))
            < static_cast<std::size_t>(3)*edge_count
        || execution_prepared_r.extent(0)
            < static_cast<std::size_t>(edge_count)
        || (dual_layer_tiled_plan_active
                ? dual_layer_workspace_neigh_types.extent(0)
                : single_layer_workspace_neigh_types.extent(0))
            < static_cast<std::size_t>(edge_count))
        throw std::invalid_argument(
            "Single-layer tiled geometry has invalid edge bounds.");

    const auto reference_xyz = execution_prepared_reference_xyz;
    const auto reference_positions = execution_prepared_reference_positions;
    const auto shifts = execution_prepared_edge_shifts;
    const auto cell = execution_prepared_cell;
    const auto edge_sources = execution_prepared_neigh_indices;
    const auto receiver_features =
        execution_prepared_receiver_feature_indices;
    const auto num_neigh = execution_prepared_num_neigh;
    const auto first_neigh = streamed_first_neigh;
    const auto neigh_types = dual_layer_tiled_plan_active
        ? dual_layer_workspace_neigh_types
        : single_layer_workspace_neigh_types;
    const auto edge_local_receivers = dual_layer_edge_local_receivers;
    const bool fill_local_receivers = dual_layer_tiled_plan_active;
    const auto displacements = execution_prepared_displacements;
    const bool fractional_geometry = factorized_prepared_geometry_fractional;
    const bool shift_geometry = factorized_prepared_geometry_uses_shifts;
    const auto direction = execution_prepared_unit_direction;
    const auto radius = execution_prepared_r;
    const auto explicit_xyz = single_layer_workspace_xyz;
    const auto explicit_positions =
        single_layer_explicit_feature_positions_device;
    const auto invalid = execution_prepared_geometry_invalid;
    const double cutoff = r_cut;
    if (explicit_host_geometry) {
        Kokkos::deep_copy(
            factorized_execution_space,
            Kokkos::subview(
                explicit_xyz,
                Kokkos::make_pair(
                    std::size_t(0), static_cast<std::size_t>(3)*edge_count)),
            Kokkos::subview(
                single_layer_explicit_xyz_host,
                Kokkos::make_pair(
                    static_cast<std::size_t>(3)*edge_begin,
                    static_cast<std::size_t>(3)*edge_end)));
        Kokkos::deep_copy(
            factorized_execution_space,
            Kokkos::subview(
                radius,
                Kokkos::make_pair(
                    std::size_t(0), static_cast<std::size_t>(edge_count))),
            Kokkos::subview(
                single_layer_explicit_r_host,
                Kokkos::make_pair(
                    static_cast<std::size_t>(edge_begin),
                    edge_end)));
    } else if (explicit_device_geometry) {
        Kokkos::deep_copy(
            factorized_execution_space,
            Kokkos::subview(
                explicit_xyz,
                Kokkos::make_pair(
                    std::size_t(0), static_cast<std::size_t>(3)*edge_count)),
            Kokkos::subview(
                single_layer_explicit_xyz_device,
                Kokkos::make_pair(
                    static_cast<std::size_t>(3)*edge_begin,
                    static_cast<std::size_t>(3)*edge_end)));
        Kokkos::deep_copy(
            factorized_execution_space,
            Kokkos::subview(
                radius,
                Kokkos::make_pair(
                    std::size_t(0), static_cast<std::size_t>(edge_count))),
            Kokkos::subview(
                single_layer_explicit_r_device,
                Kokkos::make_pair(
                    static_cast<std::size_t>(edge_begin), edge_end)));
    }
    using team_policy = Kokkos::TeamPolicy<
        decltype(factorized_execution_space)>;
    using member_type = typename team_policy::member_type;
    Kokkos::parallel_for(
        "MACEKokkos::single_layer_tiled_edge_geometry",
        team_policy(factorized_execution_space, receiver_count, Kokkos::AUTO),
        KOKKOS_LAMBDA (const member_type& team) {
            const int local_receiver = team.league_rank();
            const int receiver = receiver_begin+local_receiver;
            const int local_edge_begin = first_neigh(local_receiver);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, num_neigh(receiver)),
                [=] (const int neighbor) {
                    const int local_edge = local_edge_begin+neighbor;
                    const int edge = edge_begin+local_edge;
                    if (fill_local_receivers)
                        edge_local_receivers(local_edge) = local_receiver;
                    const std::size_t reference_offset = std::size_t(3)
                        *static_cast<std::size_t>(edge);
                    const int source = edge_sources(edge);
                    neigh_types(local_edge) = feature_types(source);
                    double vector[3];
                    if (explicit_edge_geometry) {
                        const double distance = radius(local_edge);
                        bool valid = Kokkos::isfinite(distance) && distance > 0.0;
                        for (int component=0; component<3; ++component) {
                            vector[component] = explicit_xyz(
                                3*local_edge+component);
                            valid = valid && Kokkos::isfinite(vector[component]);
                        }
                        if (!valid) {
                            Kokkos::atomic_add(&invalid(0), 1);
                            if (compact_geometry) {
                                direction(3*local_edge) = Precision(1);
                                direction(3*local_edge+1) = Precision(0);
                                direction(3*local_edge+2) = Precision(0);
                            } else {
                                explicit_xyz(3*local_edge) = cutoff;
                                explicit_xyz(3*local_edge+1) = 0.0;
                                explicit_xyz(3*local_edge+2) = 0.0;
                            }
                            radius(local_edge) = cutoff;
                            return;
                        }
                        const double scale = distance >= cutoff
                            ? cutoff/distance : 1.0;
                        for (int component=0; component<3; ++component) {
                            if (compact_geometry)
                                direction(3*local_edge+component) =
                                    static_cast<Precision>(
                                        vector[component]/distance);
                            else
                                explicit_xyz(3*local_edge+component) =
                                    scale*vector[component];
                        }
                        radius(local_edge) = distance >= cutoff ? cutoff : distance;
                        return;
                    }
                    double squared_distance = 0.0;
                    if (explicit_feature_positions) {
                        const int receiver_feature = receiver_features(receiver);
                        for (int component=0; component<3; ++component) {
                            vector[component] = explicit_positions(
                                3*source+component)-explicit_positions(
                                3*receiver_feature+component);
                            squared_distance +=
                                vector[component]*vector[component];
                        }
                    } else {
                        for (int component=0; component<3; ++component) {
                            double reference;
                            if (shift_geometry) {
                                reference = reference_positions(3*source+component)
                                    -reference_positions(3*receiver+component);
                                for (int lattice=0; lattice<3; ++lattice)
                                    reference += static_cast<double>(
                                        shifts(reference_offset+lattice))
                                        *cell(3*lattice+component);
                            } else {
                                reference = reference_xyz(
                                    reference_offset+component);
                                if (fractional_geometry) {
                                    reference = 0.0;
                                    for (int lattice=0; lattice<3; ++lattice)
                                        reference += reference_xyz(
                                            reference_offset+lattice)
                                            *cell(3*lattice+component);
                                }
                            }
                            vector[component] = reference
                                +displacements(3*source+component)
                                -displacements(3*receiver+component);
                            squared_distance +=
                                vector[component]*vector[component];
                        }
                    }
                    if (!Kokkos::isfinite(squared_distance)
                        || !(squared_distance > 0.0)) {
                        Kokkos::atomic_add(&invalid(0), 1);
                        if (compact_geometry) {
                            direction(3*local_edge) = Precision(1);
                            direction(3*local_edge+1) = Precision(0);
                            direction(3*local_edge+2) = Precision(0);
                        } else {
                            explicit_xyz(3*local_edge) = cutoff;
                            explicit_xyz(3*local_edge+1) = 0.0;
                            explicit_xyz(3*local_edge+2) = 0.0;
                        }
                        radius(local_edge) = cutoff;
                        return;
                    }
                    const double distance = Kokkos::sqrt(squared_distance);
                    const double scale = distance >= cutoff
                        ? cutoff/distance : 1.0;
                    for (int component=0; component<3; ++component) {
                        if (compact_geometry)
                            direction(3*local_edge+component) =
                                static_cast<Precision>(
                                    vector[component]/distance);
                        else
                            explicit_xyz(3*local_edge+component) =
                                scale*vector[component];
                    }
                    radius(local_edge) = distance >= cutoff ? cutoff : distance;
                });
        });
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types)
{
    return prepare_factorized_graph_device(
        num_nodes, num_nodes, node_types, num_neigh, neigh_indices, neigh_types);
}

template <typename Precision>
std::pair<std::uint64_t,std::size_t>
MACEKokkos<Precision>::prepare_periodic_factorized_graph(
    const std::span<const int> node_types,
    const std::span<const double> positions,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const double neighbor_cutoff)
{
    const auto start = std::chrono::steady_clock::now();
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        throw std::logic_error(
            "A prepared factorized graph cannot change during a distributed evaluation.");
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Native periodic graph construction requires streamed_edges='direct'.");
    if (positions.size()%3 != 0
        || node_types.size() != positions.size()/3)
        throw std::invalid_argument(
            "Native periodic graph node and position extents are inconsistent.");
    if (node_types.size()
            >= static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::length_error(
            "Native periodic graph exceeds 32-bit node indexing.");
    const int num_nodes = static_cast<int>(node_types.size());
    auto prepared_node_types = std::vector<int>(node_types.begin(), node_types.end());
    for (const int type : prepared_node_types)
        if (type < 0 || type >= static_cast<int>(atomic_numbers_host.size()))
            throw std::out_of_range(
                "Native periodic graph node type is out of range.");

    prepare_active_types(prepared_node_types);
    release_dual_layer_source_schedule();
    std::size_t graph_replacements_before =
        factorized_graph_device_replacement_count;
    auto graph = symmetrix::execution::build_periodic_neighbor_graph_kokkos(
        positions, cell, inverse_cell, neighbor_cutoff, false,
        [&] (symmetrix::execution::KokkosNeighborGraph& counted_graph) {
            validate_graph_cardinality(
                num_nodes, num_nodes, counted_graph.num_edges);
            invalidate_factorized_prepared_graph();
            const auto counted_num_neigh = Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(), counted_graph.num_neigh);
            select_low_memory_policy_for_graph(
                num_nodes, num_nodes, counted_graph.num_edges,
                std::span<const int>(
                    counted_num_neigh.data(), counted_num_neigh.extent(0)));
            graph_replacements_before =
                factorized_graph_device_replacement_count;
            reserve_factorized_graph_device_capacity(
                num_nodes, num_nodes,
                static_cast<int>(counted_graph.num_edges), 0, 0);
            const bool shift_geometry = single_layer_tiled_plan_active
                || dual_layer_tiled_plan_active;
            reserve_prepared_geometry_state_capacity(
                node_types.size(), counted_graph.num_edges, shift_geometry);
            if (counted_graph.num_edges == 0) {
                execution_prepared_neigh_indices = {};
                execution_prepared_edge_shifts = {};
                counted_graph.sources = {};
                counted_graph.shifts = {};
                counted_graph.fractional_xyz = {};
                return;
            }
            execution_prepared_neigh_indices = Kokkos::subview(
                execution_prepared_neigh_indices_storage,
                Kokkos::make_pair(std::size_t(0), counted_graph.num_edges));
            counted_graph.sources = execution_prepared_neigh_indices;
            if (shift_geometry) {
                const std::size_t shift_capacity = std::size_t(3)*std::max(
                    counted_graph.num_edges,
                    static_cast<std::size_t>(
                        std::max(0, execution_planned_edges)));
                if (execution_prepared_edge_shifts_storage.extent(0)
                        < shift_capacity) {
                    execution_prepared_edge_shifts = {};
                    execution_prepared_edge_shifts_storage = Kokkos::View<int*>(
                        Kokkos::view_alloc(
                            "Execution prepared edge shifts capacity",
                            Kokkos::WithoutInitializing),
                        shift_capacity);
                    factorized_geometry_state_allocation_count += 1;
                }
                execution_prepared_edge_shifts = Kokkos::subview(
                    execution_prepared_edge_shifts_storage,
                    Kokkos::make_pair(
                        std::size_t(0),
                        std::size_t(3)*counted_graph.num_edges));
                counted_graph.shifts = Kokkos::View<
                    int*[3],Kokkos::LayoutRight>(
                        execution_prepared_edge_shifts.data(),
                        counted_graph.num_edges);
                counted_graph.fractional_xyz = {};
            } else {
                execution_prepared_edge_shifts = {};
                counted_graph.shifts = {};
                counted_graph.fractional_xyz = Kokkos::View<
                    double*[3],Kokkos::LayoutRight>(
                    execution_prepared_reference_xyz.data(),
                    counted_graph.num_edges);
            }
        });
    validate_graph_cardinality(num_nodes, num_nodes, graph.num_edges);
    const int num_edges = static_cast<int>(graph.num_edges);
    const auto host_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), graph.num_neigh);
    auto prepared_num_neigh = std::vector<int>(num_nodes);
    auto receiver_offsets = std::vector<int>(num_nodes);
    auto chunk_edge_offsets = std::vector<int>();
    int counted_edges = 0;
    for (int receiver=0; receiver<num_nodes; ++receiver) {
        prepared_num_neigh[receiver] = host_num_neigh(receiver);
        receiver_offsets[receiver] = counted_edges;
        counted_edges += prepared_num_neigh[receiver];
    }
    if (counted_edges != num_edges)
        throw std::runtime_error(
            "Native periodic graph receiver degrees do not sum to its edge count.");

    factorized_chunk_size = select_factorized_chunk_size(
        num_nodes, num_nodes, num_edges, prepared_num_neigh);
    const int num_chunks = num_nodes/factorized_chunk_size
        +(num_nodes%factorized_chunk_size != 0);
    chunk_edge_offsets.resize(num_chunks+1, num_edges);
    for (int chunk=0; chunk<num_chunks; ++chunk)
        chunk_edge_offsets[chunk] = receiver_offsets[chunk*factorized_chunk_size];

    const bool fixed_workspace_tiled = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const bool source_schedule_required = !single_layer_readout
        && !dual_layer_tiled_plan_active;
    if (A0_scaled && r0_supports_low_memory()
        && !fixed_workspace_tiled
        && standard_r0_density_state.extent(0)
            != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(standard_r0_density_state, num_nodes);

    const auto active_prefix = [] (
        const Kokkos::View<int*>& storage,
        const std::size_t count) -> Kokkos::View<int*> {
        if (count == 0)
            return {};
        return Kokkos::subview(
            storage, Kokkos::make_pair(std::size_t(0), count));
    };
    execution_source_chunk_offsets = {};
    execution_source_edges = {};
    execution_prepared_node_types = active_prefix(
        execution_prepared_node_types_storage, node_types.size());
    execution_prepared_num_neigh = active_prefix(
        execution_prepared_num_neigh_storage, node_types.size());
    execution_prepared_neigh_indices = active_prefix(
        execution_prepared_neigh_indices_storage, graph.num_edges);
    if (fixed_workspace_tiled) {
        execution_prepared_neigh_types = {};
        execution_edge_receivers = {};
        execution_direct_source_offsets = {};
        execution_direct_source_edges = {};
        streamed_first_neigh = {};
    } else {
        execution_prepared_neigh_types = active_prefix(
            execution_prepared_neigh_types_storage, graph.num_edges);
        execution_edge_receivers = active_prefix(
            execution_edge_receivers_storage, graph.num_edges);
        if (source_schedule_required) {
            execution_direct_source_offsets = active_prefix(
                execution_direct_source_offsets_storage,
                static_cast<std::size_t>(num_nodes)+1);
            execution_direct_source_edges = active_prefix(
                execution_direct_source_edges_storage, graph.num_edges);
        } else {
            execution_direct_source_offsets = {};
            execution_direct_source_edges = {};
        }
        streamed_first_neigh = active_prefix(
            streamed_first_neigh_storage, node_types.size());
    }

    copy_int_prefix(
        factorized_execution_space, execution_prepared_node_types_storage,
        execution_prepared_node_types, prepared_node_types);
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_num_neigh,
        graph.num_neigh);
    if (!fixed_workspace_tiled) {
        Kokkos::deep_copy(
            factorized_execution_space, streamed_first_neigh,
            Kokkos::subview(
                graph.receiver_offsets,
                Kokkos::make_pair(std::size_t(0), node_types.size())));
        const auto prepared_sources = execution_prepared_neigh_indices;
        const auto prepared_types = execution_prepared_neigh_types;
        const auto prepared_nodes = execution_prepared_node_types;
        const auto edge_receivers = execution_edge_receivers;
        const auto first_neigh = streamed_first_neigh;
        const auto num_neigh_device = execution_prepared_num_neigh;
        auto direct_source_offsets = execution_direct_source_offsets;
        auto direct_source_edges = execution_direct_source_edges;
        if (source_schedule_required)
            Kokkos::deep_copy(
                factorized_execution_space, direct_source_offsets, 0);
        Kokkos::parallel_for(
            "MACEKokkos::prepare_periodic_edge_metadata",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const int receiver) {
                const int edge_begin = first_neigh(receiver);
                const int edge_end = edge_begin+num_neigh_device(receiver);
                for (int edge=edge_begin; edge<edge_end; ++edge) {
                    const int source = prepared_sources(edge);
                    edge_receivers(edge) = receiver;
                    prepared_types(edge) = prepared_nodes(source);
                    if (source_schedule_required)
                        Kokkos::atomic_fetch_add(
                            &direct_source_offsets(source+1), 1);
                }
            });
        if (source_schedule_required) {
            Kokkos::parallel_scan(
                "MACEKokkos::scan_periodic_source_offsets",
                Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                    factorized_execution_space, 0, num_nodes+1),
                KOKKOS_LAMBDA (const int source, int& update, const bool final) {
                    update += direct_source_offsets(source);
                    if (final)
                        direct_source_offsets(source) = update;
                });
            auto direct_source_cursors = Kokkos::View<int*>(
                Kokkos::view_alloc(
                    Kokkos::WithoutInitializing,
                    "Execution periodic direct source cursors"),
                static_cast<std::size_t>(num_nodes)+1);
            Kokkos::deep_copy(
                factorized_execution_space,
                direct_source_cursors, direct_source_offsets);
            Kokkos::parallel_for(
                "MACEKokkos::fill_periodic_direct_source_edges",
                Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                    factorized_execution_space, 0, num_edges),
                KOKKOS_LAMBDA (const int edge) {
                    const int source = prepared_sources(edge);
                    const int offset = Kokkos::atomic_fetch_add(
                        &direct_source_cursors(source), 1);
                    direct_source_edges(offset) = edge;
                });
            Kokkos::parallel_for(
                "MACEKokkos::sort_periodic_direct_source_edges",
                Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                    factorized_execution_space, 0, num_nodes),
                KOKKOS_LAMBDA (const int source) {
                    const int begin = direct_source_offsets(source);
                    const int end = direct_source_offsets(source+1);
                    for (int index=begin+1; index<end; ++index) {
                        const int edge = direct_source_edges(index);
                        int insertion = index;
                        while (insertion > begin
                            && direct_source_edges(insertion-1) > edge) {
                            direct_source_edges(insertion) =
                                direct_source_edges(insertion-1);
                            --insertion;
                        }
                        direct_source_edges(insertion) = edge;
                    }
                });
        }
    }

    execution_schedule_num_neigh_host = prepared_num_neigh;
    execution_schedule_neigh_indices_host.clear();
    execution_prepared_node_types_host = prepared_node_types;
    execution_prepared_num_neigh_host = prepared_num_neigh;
    execution_prepared_neigh_indices_host.clear();
    execution_prepared_neigh_types_host.clear();
    execution_receiver_offsets_host = std::move(receiver_offsets);
    execution_chunk_edge_offsets_host = std::move(chunk_edge_offsets);
    execution_schedule_num_nodes = num_nodes;
    execution_schedule_num_feature_nodes = num_nodes;
    execution_prepared_num_feature_nodes = num_nodes;
    execution_schedule_num_chunks = num_chunks;
    factorized_schedule_bytes = sizeof(int)*(
        execution_direct_source_offsets.size()
        +execution_direct_source_edges.size()+execution_edge_receivers.size()
        +streamed_first_neigh.size());
    if (dual_layer_tiled_plan_active) {
        prepare_dual_layer_tile_source_schedule_device(
            num_nodes, num_nodes, num_edges);
        prepare_dual_layer_tiled_workspace(num_nodes, num_nodes);
    } else if (single_layer_tiled_plan_active)
        prepare_single_layer_tiled_workspace(num_nodes);
    factorized_coupling_capacity_bytes = 0;
    factorized_schedule_build_count += 1;
    update_factorized_workspace_accounting();
    plan_factorized_reverse_cache();
    release_factorized_stateful_workspace();
    factorized_schedule_dirty = false;

    reserve_execution_geometry_workspace(num_edges);
    reserve_prepared_geometry_state_capacity(
        node_types.size(), graph.num_edges, fixed_workspace_tiled);
    const auto reference_positions = execution_prepared_reference_positions;
    if (fixed_workspace_tiled) {
        const auto positions_host = Kokkos::View<
            const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                positions.data(), positions.size());
        Kokkos::deep_copy(
            factorized_execution_space, reference_positions, positions_host);
    } else {
        const auto graph_fractional_positions = graph.fractional_positions;
        Kokkos::parallel_for(
            "MACEKokkos::copy_periodic_fractional_reference_positions",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, std::size_t(0),
                std::size_t(3)*static_cast<std::size_t>(num_nodes)),
            KOKKOS_LAMBDA (const std::size_t flat) {
                reference_positions(flat) = graph_fractional_positions(
                    flat/3, flat%3);
            });
    }
    if (execution_prepared_cell.extent(0) != 9) {
        execution_prepared_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared cell", Kokkos::WithoutInitializing), 9);
        execution_prepared_cell_host = Kokkos::View<double*,Kokkos::HostSpace>(
            Kokkos::view_alloc(
                "Execution prepared cell host", Kokkos::WithoutInitializing), 9);
        factorized_geometry_state_allocation_count += 2;
    }
    if (execution_prepared_inverse_cell.extent(0) != 9) {
        execution_prepared_inverse_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared inverse cell", Kokkos::WithoutInitializing), 9);
        execution_prepared_inverse_cell_host =
            Kokkos::View<double*,Kokkos::HostSpace>(
                Kokkos::view_alloc(
                    "Execution prepared inverse cell host",
                    Kokkos::WithoutInitializing), 9);
        factorized_geometry_state_allocation_count += 2;
    }
    std::copy(cell.begin(), cell.end(), execution_prepared_cell_host.data());
    std::copy(
        inverse_cell.begin(), inverse_cell.end(),
        execution_prepared_inverse_cell_host.data());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_cell,
        execution_prepared_cell_host);
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_inverse_cell,
        execution_prepared_inverse_cell_host);
    if (execution_prepared_pbc.extent(0) != 3) {
        execution_prepared_pbc = Kokkos::View<int*>(
            "Execution prepared pbc", 3);
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(factorized_execution_space, execution_prepared_pbc, 1);
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }

    factorized_execution_space.fence(
        "Native periodic factorized graph preparation");
    factorized_preparation_fence_count += 1;
    if (factorized_graph_device_replacement_count == graph_replacements_before)
        factorized_graph_device_update_count += 1;
    factorized_graph_generation_counter += 1;
    if (factorized_graph_generation_counter == 0)
        factorized_graph_generation_counter += 1;
    factorized_prepared_graph_generation = factorized_graph_generation_counter;
    factorized_prepared_geometry_graph_generation =
        factorized_prepared_graph_generation;
    factorized_prepared_geometry_fractional = !fixed_workspace_tiled;
    factorized_prepared_geometry_uses_shifts = fixed_workspace_tiled;
    factorized_prepared_graph_count += 1;
    factorized_fractional_geometry_preparation_count += 1;
    factorized_fractional_geometry_initialization_bytes +=
        sizeof(double)*(3*node_types.size()+18)
        +(fixed_workspace_tiled
            ? sizeof(int)*3*graph.num_edges
            : sizeof(double)*3*graph.num_edges)
        +3*sizeof(int);
    factorized_last_graph_prepare_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    return {factorized_prepared_graph_generation, graph.num_edges};
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types)
{
    const auto host_node_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_types);
    const auto host_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_neigh);
    const auto host_neigh_indices = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_indices);
    const auto host_neigh_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_types);
    return prepare_factorized_graph(
        num_receivers, num_feature_nodes,
        std::span<const int>(host_node_types.data(), host_node_types.extent(0)),
        std::span<const int>(host_num_neigh.data(), host_num_neigh.extent(0)),
        std::span<const int>(
            host_neigh_indices.data(), host_neigh_indices.extent(0)),
        std::span<const int>(
            host_neigh_types.data(), host_neigh_types.extent(0)));
}

template <typename Precision>
std::uint64_t MACEKokkos<Precision>::prepare_factorized_graph_device(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const int*> receiver_feature_indices,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> feature_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types)
{
    const auto host_receiver_feature_indices =
        Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), receiver_feature_indices);
    const auto host_node_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), node_types);
    const auto host_feature_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), feature_types);
    const auto host_num_neigh = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), num_neigh);
    const auto host_neigh_indices = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_indices);
    const auto host_neigh_types = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), neigh_types);
    return prepare_factorized_graph(
        num_receivers, num_feature_nodes,
        std::span<const int>(
            host_receiver_feature_indices.data(),
            host_receiver_feature_indices.extent(0)),
        std::span<const int>(host_node_types.data(), host_node_types.extent(0)),
        std::span<const int>(
            host_feature_types.data(), host_feature_types.extent(0)),
        std::span<const int>(host_num_neigh.data(), host_num_neigh.extent(0)),
        std::span<const int>(
            host_neigh_indices.data(), host_neigh_indices.extent(0)),
        std::span<const int>(
            host_neigh_types.data(), host_neigh_types.extent(0)));
}

template <typename Precision>
void MACEKokkos<Precision>::reserve_prepared_geometry_state_capacity(
    const std::size_t num_nodes,
    const std::size_t num_edges,
    const bool shift_geometry)
{
    const std::size_t node_capacity = std::max(
        num_nodes,
        static_cast<std::size_t>(std::max(0, execution_planned_feature_nodes)));
    const std::size_t edge_capacity = std::max(
        num_edges,
        static_cast<std::size_t>(std::max(0, execution_planned_edges)));
    // Graph cardinality validation bounds both capacities by signed int max.
    const std::size_t node_scalars = std::size_t(3)*node_capacity;
    const std::size_t edge_scalars = std::size_t(3)*edge_capacity;
    const bool fits = execution_prepared_reference_positions_storage.extent(0)
            >= node_scalars
        && execution_prepared_displacements_storage.extent(0) >= node_scalars
        && (shift_geometry
            ? execution_prepared_positions_storage.extent(0) == 0
                && execution_prepared_reference_xyz_storage.extent(0) == 0
            : execution_prepared_positions_storage.extent(0) >= node_scalars
                && execution_prepared_reference_xyz_storage.extent(0)
                    >= edge_scalars);
    if (!fits) {
        factorized_execution_space.fence(
            "Replace prepared geometry state capacity");
        execution_prepared_positions = {};
        execution_prepared_reference_positions = {};
        execution_prepared_reference_xyz = {};
        execution_prepared_displacements = {};
        execution_prepared_positions_storage = {};
        execution_prepared_reference_positions_storage = {};
        execution_prepared_reference_xyz_storage = {};
        execution_prepared_displacements_storage = {};
        const auto allocate = [] (const char* label, const std::size_t count) {
            return Kokkos::View<double*>(
                Kokkos::view_alloc(
                    std::string(label), Kokkos::WithoutInitializing), count);
        };
        if (!shift_geometry)
            execution_prepared_positions_storage = allocate(
                "Execution prepared positions capacity", node_scalars);
        execution_prepared_reference_positions_storage = allocate(
            "Execution prepared reference positions capacity", node_scalars);
        if (!shift_geometry)
            execution_prepared_reference_xyz_storage = allocate(
                "Execution prepared reference xyz capacity", edge_scalars);
        execution_prepared_displacements_storage = allocate(
            "Execution prepared displacements capacity", node_scalars);
        factorized_geometry_state_allocation_count += shift_geometry ? 2 : 4;
    }
    const auto prefix = [] (
        const Kokkos::View<double*>& storage,
        const std::size_t count) -> Kokkos::View<double*> {
        if (count == 0)
            return {};
        return Kokkos::subview(
            storage, Kokkos::make_pair(std::size_t(0), count));
    };
    execution_prepared_positions = shift_geometry ? Kokkos::View<double*>() : prefix(
        execution_prepared_positions_storage, 3*num_nodes);
    execution_prepared_reference_positions = prefix(
        execution_prepared_reference_positions_storage, 3*num_nodes);
    execution_prepared_reference_xyz = shift_geometry ? Kokkos::View<double*>() : prefix(
        execution_prepared_reference_xyz_storage, 3*num_edges);
    execution_prepared_displacements = prefix(
        execution_prepared_displacements_storage, 3*num_nodes);
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> reference_positions,
    const std::span<const double> reference_xyz,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::span<const int> pbc)
{
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 geometry requires the current prepared graph token.");
    if (execution_prepared_num_feature_nodes
        != static_cast<int>(execution_prepared_node_types.extent(0)))
        throw std::invalid_argument(
            "Prepared position geometry does not support distributed factorized graphs.");
    const std::size_t num_nodes = execution_prepared_node_types.extent(0);
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (reference_positions.size() != 3*num_nodes
        || reference_xyz.size() != 3*num_edges
        || cell.size() != 9 || inverse_cell.size() != 9 || pbc.size() != 3)
        throw std::invalid_argument(
            "Execution R1 prepared geometry extents are inconsistent.");
    if (!std::all_of(
            reference_positions.begin(), reference_positions.end(),
            [] (const double value) { return std::isfinite(value); })
        || !std::all_of(
            reference_xyz.begin(), reference_xyz.end(),
            [] (const double value) { return std::isfinite(value); })
        || !std::all_of(
            cell.begin(), cell.end(),
            [] (const double value) { return std::isfinite(value); })
        || !std::all_of(
            inverse_cell.begin(), inverse_cell.end(),
            [] (const double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "Execution R1 prepared geometry contains a non-finite value.");

    reserve_prepared_geometry_state_capacity(num_nodes, num_edges);
    const auto copy_double = [&] (
        const std::span<const double> source,
        Kokkos::View<double*>& destination) {
        const auto host = Kokkos::View<
            const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                source.data(), source.size());
        if (destination.extent(0) != source.size())
            throw std::logic_error(
                "Prepared geometry active prefix has an invalid extent.");
        Kokkos::deep_copy(factorized_execution_space, destination, host);
    };
    copy_double(
        reference_positions,
        execution_prepared_reference_positions);
    copy_double(
        reference_xyz,
        execution_prepared_reference_xyz);
    if (execution_prepared_cell.extent(0) != cell.size()) {
        execution_prepared_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared cell", Kokkos::WithoutInitializing),
            cell.size());
        execution_prepared_cell_host =
            Kokkos::View<double*,Kokkos::HostSpace>(
                Kokkos::view_alloc(
                    "Execution prepared cell host", Kokkos::WithoutInitializing),
                cell.size());
        factorized_geometry_state_allocation_count += 2;
    }
    if (execution_prepared_inverse_cell.extent(0) != inverse_cell.size()) {
        execution_prepared_inverse_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared inverse cell", Kokkos::WithoutInitializing),
            inverse_cell.size());
        execution_prepared_inverse_cell_host =
            Kokkos::View<double*,Kokkos::HostSpace>(
                Kokkos::view_alloc(
                    "Execution prepared inverse cell host",
                    Kokkos::WithoutInitializing),
                inverse_cell.size());
        factorized_geometry_state_allocation_count += 2;
    }
    std::copy(cell.begin(), cell.end(), execution_prepared_cell_host.data());
    std::copy(
        inverse_cell.begin(), inverse_cell.end(),
        execution_prepared_inverse_cell_host.data());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_cell,
        execution_prepared_cell_host);
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_inverse_cell,
        execution_prepared_inverse_cell_host);
    const auto pbc_host = Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            pbc.data(), pbc.size());
    if (execution_prepared_pbc.extent(0) != pbc.size()) {
        execution_prepared_pbc = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Execution prepared pbc"),
            pbc.size());
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(factorized_execution_space, execution_prepared_pbc, pbc_host);
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }
    factorized_execution_space.fence("Execution R1 geometry preparation");
    factorized_prepared_geometry_graph_generation = graph_generation;
    factorized_prepared_geometry_fractional = false;
    factorized_prepared_geometry_uses_shifts = false;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_shift_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> reference_positions,
    const std::span<const int> shifts,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::span<const int> pbc)
{
    if (!single_layer_tiled_plan_active && !dual_layer_tiled_plan_active)
        throw std::invalid_argument(
            "Shift geometry requires a fixed-workspace tiled plan.");
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Shift geometry requires the current prepared graph token.");
    if (execution_prepared_num_feature_nodes
        != static_cast<int>(execution_prepared_node_types.extent(0)))
        throw std::invalid_argument(
            "Shift geometry does not support distributed factorized graphs.");
    const std::size_t num_nodes = execution_prepared_node_types.extent(0);
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (reference_positions.size() != 3*num_nodes
        || shifts.size() != 3*num_edges
        || cell.size() != 9 || inverse_cell.size() != 9 || pbc.size() != 3)
        throw std::invalid_argument(
            "Shift geometry extents are inconsistent.");
    if (!std::all_of(
            reference_positions.begin(), reference_positions.end(),
            [] (const double value) { return std::isfinite(value); })
        || !std::all_of(
            cell.begin(), cell.end(),
            [] (const double value) { return std::isfinite(value); })
        || !std::all_of(
            inverse_cell.begin(), inverse_cell.end(),
            [] (const double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "Shift geometry contains a non-finite value.");

    reserve_prepared_geometry_state_capacity(num_nodes, num_edges, true);
    const auto reference_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            reference_positions.data(), reference_positions.size());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_reference_positions,
        reference_host);

    const std::size_t shift_capacity = std::size_t(3)*std::max(
        num_edges,
        static_cast<std::size_t>(std::max(0, execution_planned_edges)));
    if (execution_prepared_edge_shifts_storage.extent(0) < shift_capacity) {
        factorized_execution_space.fence(
            "Replace prepared edge shift capacity");
        execution_prepared_edge_shifts = {};
        execution_prepared_edge_shifts_storage = Kokkos::View<int*>(
            Kokkos::view_alloc(
                "Execution prepared edge shifts capacity",
                Kokkos::WithoutInitializing),
            shift_capacity);
        factorized_geometry_state_allocation_count += 1;
    }
    execution_prepared_edge_shifts = Kokkos::subview(
        execution_prepared_edge_shifts_storage,
        Kokkos::make_pair(std::size_t(0), shifts.size()));
    const auto shifts_host = Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            shifts.data(), shifts.size());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_edge_shifts,
        shifts_host);

    if (execution_prepared_cell.extent(0) != cell.size()) {
        execution_prepared_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared cell", Kokkos::WithoutInitializing),
            cell.size());
        execution_prepared_cell_host = Kokkos::View<double*,Kokkos::HostSpace>(
            Kokkos::view_alloc(
                "Execution prepared cell host", Kokkos::WithoutInitializing),
            cell.size());
        factorized_geometry_state_allocation_count += 2;
    }
    if (execution_prepared_inverse_cell.extent(0) != inverse_cell.size()) {
        execution_prepared_inverse_cell = Kokkos::View<double*>(
            Kokkos::view_alloc(
                "Execution prepared inverse cell", Kokkos::WithoutInitializing),
            inverse_cell.size());
        execution_prepared_inverse_cell_host =
            Kokkos::View<double*,Kokkos::HostSpace>(
                Kokkos::view_alloc(
                    "Execution prepared inverse cell host",
                    Kokkos::WithoutInitializing),
                inverse_cell.size());
        factorized_geometry_state_allocation_count += 2;
    }
    std::copy(cell.begin(), cell.end(), execution_prepared_cell_host.data());
    std::copy(
        inverse_cell.begin(), inverse_cell.end(),
        execution_prepared_inverse_cell_host.data());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_cell,
        execution_prepared_cell_host);
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_inverse_cell,
        execution_prepared_inverse_cell_host);
    const auto pbc_host = Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            pbc.data(), pbc.size());
    if (execution_prepared_pbc.extent(0) != pbc.size()) {
        execution_prepared_pbc = Kokkos::View<int*>(
            Kokkos::view_alloc(
                Kokkos::WithoutInitializing, "Execution prepared pbc"),
            pbc.size());
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_pbc, pbc_host);
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }
    factorized_execution_space.fence("Execution shift geometry preparation");
    factorized_prepared_geometry_graph_generation = graph_generation;
    factorized_prepared_geometry_fractional = false;
    factorized_prepared_geometry_uses_shifts = true;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_fractional_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> fractional_positions,
    const std::span<const double> fractional_xyz,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell,
    const std::span<const int> pbc)
{
    if (cell.size() != 9 || inverse_cell.size() != 9 || pbc.size() != 3)
        throw std::invalid_argument(
            "Fractional Execution R1 geometry requires 3x3 cell matrices and "
            "three PBC flags.");
    if (!std::all_of(pbc.begin(), pbc.end(), [] (const int value) {
            return value != 0;
        }))
        throw std::invalid_argument(
            "Fractional Execution R1 geometry requires a fully periodic cell.");
    const double determinant =
        cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
        -cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
        +cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
    if (!std::isfinite(determinant) || determinant == 0.0)
        throw std::invalid_argument(
            "Fractional Execution R1 cell matrix must be invertible.");
    prepare_factorized_geometry(
        graph_generation, fractional_positions, fractional_xyz, cell,
        inverse_cell, pbc);
    factorized_prepared_geometry_fractional = true;
    factorized_fractional_geometry_preparation_count += 1;
    factorized_fractional_geometry_initialization_bytes +=
        sizeof(double)*(fractional_positions.size()+fractional_xyz.size()
            +cell.size()+inverse_cell.size())
        +sizeof(int)*pbc.size();
}

template <typename Precision>
void MACEKokkos<Precision>::update_factorized_cell(
    const std::uint64_t graph_generation,
    const std::span<const double> cell,
    const std::span<const double> inverse_cell)
{
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || graph_generation != factorized_prepared_geometry_graph_generation
        || factorized_schedule_dirty
        || (!factorized_prepared_geometry_fractional
            && !factorized_prepared_geometry_uses_shifts))
        throw std::invalid_argument(
            "Prepared Execution R1 cell update requires current graph and "
            "fractional or shift geometry tokens.");
    if (cell.size() != 9 || inverse_cell.size() != 9)
        throw std::invalid_argument(
            "Prepared Execution R1 cell matrices must contain nine values.");
    if (!std::all_of(cell.begin(), cell.end(), [] (const double value) {
            return std::isfinite(value);
        }) || !std::all_of(
            inverse_cell.begin(), inverse_cell.end(), [] (const double value) {
                return std::isfinite(value);
            }))
        throw std::invalid_argument(
            "Prepared Execution R1 cell matrices contain a non-finite value.");
    const double determinant =
        cell[0]*(cell[4]*cell[8]-cell[5]*cell[7])
        -cell[1]*(cell[3]*cell[8]-cell[5]*cell[6])
        +cell[2]*(cell[3]*cell[7]-cell[4]*cell[6]);
    if (!std::isfinite(determinant) || determinant == 0.0)
        throw std::invalid_argument(
            "Prepared Execution R1 cell matrix must be invertible.");

    std::copy(cell.begin(), cell.end(), execution_prepared_cell_host.data());
    std::copy(
        inverse_cell.begin(), inverse_cell.end(),
        execution_prepared_inverse_cell_host.data());
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_cell,
        execution_prepared_cell_host);
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_inverse_cell,
        execution_prepared_inverse_cell_host);
    factorized_cell_update_count += 1;
    factorized_cell_update_bytes += sizeof(double)*(cell.size()+inverse_cell.size());
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_factorized(
    const std::uint64_t graph_generation,
    const std::span<const double> xyz,
    const std::span<const double> r)
{
    // Match the legacy prepared entry point: even a rejected evaluation
    // invalidates any operator benchmark tied to the preceding coordinates.
    begin_factorized_production_evaluation();
    if (!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "An execution graph token requires streamed_edges='direct'.");
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 graph token is stale; prepare the graph again.");
    if (execution_prepared_num_feature_nodes
        != static_cast<int>(execution_prepared_node_types.extent(0)))
        throw std::invalid_argument(
            "The monolithic factorized evaluator does not accept distributed graphs.");

    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (r.size() != num_edges || xyz.size() != 3*num_edges)
        throw std::invalid_argument(
            "Execution R1 prepared coordinates do not match the graph extents.");
    reserve_execution_geometry_workspace(static_cast<int>(num_edges));

    if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
        const std::size_t num_nodes = execution_prepared_node_types.extent(0);
        if (execution_prepared_geometry_invalid.extent(0) != 1) {
            execution_prepared_geometry_invalid = Kokkos::View<int*>(
                "Execution prepared geometry invalid", 1);
            factorized_geometry_state_allocation_count += 1;
        }
        Kokkos::deep_copy(
            factorized_execution_space, execution_prepared_geometry_invalid, 0);
        single_layer_explicit_xyz_host = decltype(single_layer_explicit_xyz_host)(
            xyz.data(), xyz.size());
        single_layer_explicit_r_host = decltype(single_layer_explicit_r_host)(
            r.data(), r.size());
        factorized_prepared_geometry_graph_generation = 0;
        factorized_prepared_geometry_fractional = false;
        factorized_prepared_geometry_uses_shifts = false;
        compact_edge_geometry_active =
            edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
        try {
            compute_node_energies_forces(
                static_cast<int>(num_nodes),
                Kokkos::View<const int*>(), Kokkos::View<const int*>(),
                Kokkos::View<const int*>(), Kokkos::View<const int*>(),
                Kokkos::View<const double*>(), execution_prepared_r,
                graph_generation);
        } catch (...) {
            compact_edge_geometry_active = false;
            const auto failure = std::current_exception();
            factorized_execution_space.fence(
                "Execution R1 failed tiled geometry input lifetime");
            factorized_evaluation_fence_count += 1;
            single_layer_explicit_xyz_host = {};
            single_layer_explicit_r_host = {};
            std::rethrow_exception(failure);
        }
        compact_edge_geometry_active = false;
        single_layer_explicit_xyz_host = {};
        single_layer_explicit_r_host = {};
        execution_geometry_copy_count += 2;
        validate_prepared_factorized_positions_geometry();
        return;
    }

    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    const auto xyz_device = execution_prepared_xyz;
    const auto direction_device = execution_prepared_unit_direction;
    const auto r_device = Kokkos::subview(
        execution_prepared_r, Kokkos::make_pair(std::size_t(0), r.size()));
    const auto xyz_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            xyz.data(), xyz.size());
    const auto r_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            r.data(), r.size());
    std::vector<Precision> direction_storage;
    try {
        if (compact_geometry) {
            direction_storage.resize(xyz.size());
            for (std::size_t edge=0; edge<num_edges; ++edge)
                for (int component=0; component<3; ++component)
                    direction_storage[3*edge+component] =
                        static_cast<Precision>(
                            xyz[3*edge+component]/r[edge]);
            const auto direction_host = Kokkos::View<
                const Precision*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                    direction_storage.data(), direction_storage.size());
            Kokkos::deep_copy(
                factorized_execution_space,
                Kokkos::subview(
                    direction_device,
                    Kokkos::make_pair(std::size_t(0), xyz.size())),
                direction_host);
        } else {
            Kokkos::deep_copy(
                factorized_execution_space,
                Kokkos::subview(
                    xyz_device,
                    Kokkos::make_pair(std::size_t(0), xyz.size())),
                xyz_host);
        }
        Kokkos::deep_copy(factorized_execution_space, r_device, r_host);
        execution_geometry_copy_count += 2;
        compact_edge_geometry_active = compact_geometry;
        compute_node_energies_forces(
            static_cast<int>(execution_prepared_node_types.extent(0)),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            xyz_device,
            r_device,
            graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        const auto failure = std::current_exception();
        factorized_execution_space.fence(
            "Execution R1 failed evaluation host input lifetime");
        factorized_evaluation_fence_count += 1;
        std::rethrow_exception(failure);
    }
    compact_edge_geometry_active = false;
}

template <typename Precision>
void MACEKokkos<Precision>::update_prepared_factorized_positions_geometry(
    const std::uint64_t graph_generation,
    const std::span<const double> positions)
{
    begin_factorized_production_evaluation();
    if (!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "Prepared Execution R1 positions require streamed_edges='factorized'.");
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || graph_generation != factorized_prepared_geometry_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 positions require current graph and geometry tokens.");
    if (execution_prepared_num_feature_nodes
        != static_cast<int>(execution_prepared_node_types.extent(0)))
        throw std::invalid_argument(
            "Prepared positions do not support distributed factorized graphs.");
    const int num_nodes = static_cast<int>(execution_prepared_node_types.extent(0));
    const int num_edges = static_cast<int>(execution_prepared_neigh_indices.extent(0));
    if (positions.size() != 3*static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "Execution R1 prepared positions do not match the graph extent.");
    if (!std::all_of(
            positions.begin(), positions.end(),
            [] (const double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "Execution R1 prepared positions contain a non-finite value.");

    const auto positions_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            positions.data(), positions.size());
    if (factorized_prepared_geometry_uses_shifts)
        Kokkos::deep_copy(
            factorized_execution_space, execution_prepared_displacements,
            positions_host);
    else
        Kokkos::deep_copy(
            factorized_execution_space, execution_prepared_positions,
            positions_host);
    execution_geometry_copy_count += 1;
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_geometry_invalid, 0);

    const auto current_positions = factorized_prepared_geometry_uses_shifts
        ? execution_prepared_displacements : execution_prepared_positions;
    const auto reference_positions = execution_prepared_reference_positions;
    const auto reference_xyz = execution_prepared_reference_xyz;
    const auto cell = execution_prepared_cell;
    const auto inverse_cell = execution_prepared_inverse_cell;
    const auto pbc = execution_prepared_pbc;
    const auto edge_receivers = execution_edge_receivers;
    const auto edge_sources = execution_prepared_neigh_indices;
    const bool fractional_geometry = factorized_prepared_geometry_fractional;
    auto displacements = execution_prepared_displacements;
    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    auto xyz = execution_prepared_xyz;
    auto direction = execution_prepared_unit_direction;
    const bool fixed_workspace_tiled = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const std::size_t geometry_edges = fixed_workspace_tiled
        ? static_cast<std::size_t>(dual_layer_tiled_plan_active
            ? dual_layer_workspace_planned_edge_capacity
            : single_layer_workspace_planned_edge_capacity)
        : static_cast<std::size_t>(num_edges);
    auto r = Kokkos::subview(
        execution_prepared_r,
        Kokkos::make_pair(std::size_t(0), geometry_edges));
    auto invalid = execution_prepared_geometry_invalid;
    const double cutoff = r_cut;

    Kokkos::parallel_for(
        "MACEKokkos::prepared_atom_displacements",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space, 0, num_nodes),
        KOKKOS_LAMBDA (const int atom) {
            double delta[3];
            double fractional[3];
            for (int component=0; component<3; ++component) {
                double reference = reference_positions(3*atom+component);
                if (fractional_geometry) {
                    reference = 0.0;
                    for (int lattice=0; lattice<3; ++lattice)
                        reference += reference_positions(3*atom+lattice)
                            *cell(3*lattice+component);
                }
                delta[component] = current_positions(3*atom+component)-reference;
            }
            if (pbc(0) == 0 && pbc(1) == 0 && pbc(2) == 0) {
                for (int component=0; component<3; ++component)
                    displacements(3*atom+component) = delta[component];
                return;
            }
            for (int lattice=0; lattice<3; ++lattice) {
                fractional[lattice] = 0.0;
                for (int component=0; component<3; ++component)
                    fractional[lattice] +=
                        delta[component]*inverse_cell(3*component+lattice);
                if (pbc(lattice) != 0)
                    fractional[lattice] -= Kokkos::floor(
                        fractional[lattice]+0.5);
            }
            for (int component=0; component<3; ++component) {
                double value = 0.0;
                for (int lattice=0; lattice<3; ++lattice)
                    value += fractional[lattice]*cell(3*lattice+component);
                displacements(3*atom+component) = value;
            }
        });
    if (fixed_workspace_tiled) {
        factorized_prepared_geometry_update_count += 1;
        return;
    }
    Kokkos::parallel_for(
        "MACEKokkos::prepared_edge_geometry",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space, 0, num_edges),
        KOKKOS_LAMBDA (const int edge) {
            const int receiver = edge_receivers(edge);
            const int source = edge_sources(edge);
            double vector[3];
            double squared_distance = 0.0;
            for (int component=0; component<3; ++component) {
                double reference = reference_xyz(3*edge+component);
                if (fractional_geometry) {
                    reference = 0.0;
                    for (int lattice=0; lattice<3; ++lattice)
                        reference += reference_xyz(3*edge+lattice)
                            *cell(3*lattice+component);
                }
                vector[component] = reference+displacements(3*source+component)
                    -displacements(3*receiver+component);
                squared_distance += vector[component]*vector[component];
            }
            if (!Kokkos::isfinite(squared_distance) || !(squared_distance > 0.0)) {
                Kokkos::atomic_add(&invalid(0), 1);
                if (compact_geometry) {
                    direction(3*edge) = Precision(1);
                    direction(3*edge+1) = Precision(0);
                    direction(3*edge+2) = Precision(0);
                } else {
                    xyz(3*edge) = cutoff;
                    xyz(3*edge+1) = 0.0;
                    xyz(3*edge+2) = 0.0;
                }
                r(edge) = cutoff;
                return;
            }
            const double distance = Kokkos::sqrt(squared_distance);
            const double scale = distance >= cutoff ? cutoff/distance : 1.0;
            for (int component=0; component<3; ++component) {
                if (compact_geometry)
                    direction(3*edge+component) = static_cast<Precision>(
                        vector[component]/distance);
                else
                    xyz(3*edge+component) = scale*vector[component];
            }
            r(edge) = distance >= cutoff ? cutoff : distance;
        });
    factorized_prepared_geometry_update_count += 1;
}

template <typename Precision>
void MACEKokkos<Precision>::validate_prepared_factorized_positions_geometry() const
{
    const auto invalid_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), execution_prepared_geometry_invalid);
    if (invalid_host(0) != 0)
        throw std::invalid_argument(
            "Execution R1 prepared positions produced an invalid edge geometry.");
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_factorized_positions(
    const std::uint64_t graph_generation,
    const std::span<const double> positions)
{
    update_prepared_factorized_positions_geometry(graph_generation, positions);
    const int num_nodes = static_cast<int>(execution_prepared_node_types.extent(0));
    const int num_edges = static_cast<int>(execution_prepared_neigh_indices.extent(0));
    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    const auto xyz = execution_prepared_xyz;
    const bool fixed_workspace_tiled = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const std::size_t geometry_edges = fixed_workspace_tiled
        ? static_cast<std::size_t>(dual_layer_tiled_plan_active
            ? dual_layer_workspace_planned_edge_capacity
            : single_layer_workspace_planned_edge_capacity)
        : static_cast<std::size_t>(num_edges);
    const auto r = Kokkos::subview(
        execution_prepared_r,
        Kokkos::make_pair(std::size_t(0), geometry_edges));

    compact_edge_geometry_active = compact_geometry;
    try {
        compute_node_energies_forces(
            num_nodes,
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            xyz, r, graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        throw;
    }
    compact_edge_geometry_active = false;
    validate_prepared_factorized_positions_geometry();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_factorized_positions_field(
    const std::uint64_t graph_generation,
    const std::span<const double> positions,
    const std::span<const double> electric_field)
{
    if (!has_field_coupling)
        throw std::invalid_argument(
            "Prepared Execution field evaluation requires field coupling.");
    const std::size_t num_nodes = execution_prepared_node_types.extent(0);
    if (electric_field.size() != 3 && electric_field.size() != 3*num_nodes)
        throw std::invalid_argument(
            "Prepared MACEField electric_field must have shape (3,) or "
            "(num_nodes, 3).");
    if (!std::all_of(
            electric_field.begin(), electric_field.end(),
            [] (const double value) { return std::isfinite(value); }))
        throw std::invalid_argument(
            "Prepared MACEField electric_field contains a non-finite value.");

    update_prepared_factorized_positions_geometry(graph_generation, positions);
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (execution_prepared_electric_field.extent(0) < electric_field.size())
        Kokkos::realloc(execution_prepared_electric_field, electric_field.size());
    const auto field_device = Kokkos::subview(
        execution_prepared_electric_field,
        Kokkos::make_pair(std::size_t(0), electric_field.size()));
    const auto field_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            electric_field.data(), electric_field.size());
    Kokkos::deep_copy(factorized_execution_space, field_device, field_host);
    execution_geometry_copy_count += 1;

    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    const auto xyz = execution_prepared_xyz;
    const auto r = Kokkos::subview(
        execution_prepared_r,
        Kokkos::make_pair(std::size_t(0), num_edges));
    compact_edge_geometry_active = compact_geometry;
    try {
        compute_node_energies_forces_field(
            static_cast<int>(num_nodes),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            xyz, r, field_device, graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        const auto failure = std::current_exception();
        factorized_execution_space.fence(
            "Execution R1 failed position field input lifetime");
        factorized_evaluation_fence_count += 1;
        std::rethrow_exception(failure);
    }
    compact_edge_geometry_active = false;
    validate_prepared_factorized_positions_geometry();
}

template <typename Precision>
void MACEKokkos<Precision>::compute_prepared_factorized_field(
    const std::uint64_t graph_generation,
    const std::span<const double> xyz,
    const std::span<const double> r,
    const std::span<const double> electric_field)
{
    begin_factorized_production_evaluation();
    if (!has_field_coupling)
        throw std::invalid_argument(
            "Prepared Execution field evaluation requires field coupling.");
    if (!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "An execution graph token requires streamed_edges='direct' or "
            "'direct'.");
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Execution R1 graph token is stale; prepare the graph again.");

    const std::size_t num_nodes = execution_prepared_node_types.extent(0);
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (r.size() != num_edges || xyz.size() != 3*num_edges)
        throw std::invalid_argument(
            "Execution R1 prepared coordinates do not match the graph extents.");
    if (electric_field.size() != 3
        && electric_field.size() != 3*static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "Prepared MACEField electric_field must have shape (3,) or "
            "(num_nodes, 3).");
    reserve_execution_geometry_workspace(static_cast<int>(num_edges));
    if (execution_prepared_electric_field.extent(0) < electric_field.size())
        Kokkos::realloc(execution_prepared_electric_field, electric_field.size());

    const bool compact_geometry =
        edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64;
    const auto xyz_device = execution_prepared_xyz;
    const auto direction_device = execution_prepared_unit_direction;
    const auto r_device = Kokkos::subview(
        execution_prepared_r, Kokkos::make_pair(std::size_t(0), r.size()));
    const auto field_device = Kokkos::subview(
        execution_prepared_electric_field,
        Kokkos::make_pair(std::size_t(0), electric_field.size()));
    const auto xyz_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            xyz.data(), xyz.size());
    const auto r_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            r.data(), r.size());
    const auto field_host = Kokkos::View<
        const double*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            electric_field.data(), electric_field.size());
    std::vector<Precision> direction_storage;
    try {
        if (compact_geometry) {
            direction_storage.resize(xyz.size());
            for (std::size_t edge=0; edge<num_edges; ++edge)
                for (int component=0; component<3; ++component)
                    direction_storage[3*edge+component] =
                        static_cast<Precision>(
                            xyz[3*edge+component]/r[edge]);
            const auto direction_host = Kokkos::View<
                const Precision*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                    direction_storage.data(), direction_storage.size());
            Kokkos::deep_copy(
                factorized_execution_space,
                Kokkos::subview(
                    direction_device,
                    Kokkos::make_pair(std::size_t(0), xyz.size())),
                direction_host);
        } else {
            Kokkos::deep_copy(
                factorized_execution_space,
                Kokkos::subview(
                    xyz_device,
                    Kokkos::make_pair(std::size_t(0), xyz.size())),
                xyz_host);
        }
        Kokkos::deep_copy(factorized_execution_space, r_device, r_host);
        Kokkos::deep_copy(factorized_execution_space, field_device, field_host);
        execution_geometry_copy_count += 3;
        compact_edge_geometry_active = compact_geometry;
        compute_node_energies_forces_field(
            static_cast<int>(num_nodes),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            Kokkos::View<const int*>(),
            xyz_device,
            r_device,
            field_device,
            graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        const auto failure = std::current_exception();
        factorized_execution_space.fence(
            "Execution R1 failed field evaluation host input lifetime");
        factorized_evaluation_fence_count += 1;
        std::rethrow_exception(failure);
    }
    compact_edge_geometry_active = false;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_dual_layer_tile_source_schedule_host(
    const int num_receivers,
    const int num_feature_nodes,
    const std::vector<int>& num_neigh,
    const std::vector<int>& neigh_indices)
{
    if (!dual_layer_tiled_plan_active)
        return;
    dual_layer_source_schedule_preparation_explicit_scratch_bytes = 0;
    if (num_receivers <= 0 || num_feature_nodes < num_receivers
        || num_neigh.size() != static_cast<std::size_t>(num_receivers))
        throw std::invalid_argument(
            "Dual-layer tile-source schedule requires a valid distributed graph.");
    const int capacity = dual_layer_workspace_active_capacity;
    if (capacity <= 0)
        throw std::logic_error(
            "Dual-layer tile-source schedule has no receiver capacity.");

    const int num_tiles = num_receivers/capacity
        +(num_receivers%capacity != 0);
    std::vector<int> tile_segment_offsets;
    std::vector<int> segment_source_ids;
    std::vector<int> segment_edge_offsets;
    std::vector<int> source_edges;
    tile_segment_offsets.reserve(static_cast<std::size_t>(num_tiles)+1);
    source_edges.reserve(neigh_indices.size());
    tile_segment_offsets.push_back(0);

    std::size_t global_edge = 0;
    for (int tile=0; tile<num_tiles; ++tile) {
        const int receiver_begin = tile*capacity;
        const int receiver_end = receiver_begin
            +std::min(capacity, num_receivers-receiver_begin);
        const std::size_t tile_edge_begin = global_edge;
        std::vector<std::pair<int,int>> ordered_edges;
        for (int receiver=receiver_begin; receiver<receiver_end; ++receiver) {
            for (int offset=0; offset<num_neigh[receiver]; ++offset) {
                if (global_edge >= neigh_indices.size())
                    throw std::runtime_error(
                        "Dual-layer tile-source schedule exceeded its edge extent.");
                const int local_edge = static_cast<int>(global_edge-tile_edge_begin);
                ordered_edges.emplace_back(neigh_indices[global_edge], local_edge);
                ++global_edge;
            }
        }
        std::stable_sort(
            ordered_edges.begin(), ordered_edges.end(),
            [] (const auto& left, const auto& right) {
                return left.first < right.first;
            });
        int previous_source = -1;
        for (const auto [source, local_edge] : ordered_edges) {
            if (source < 0 || source >= num_feature_nodes)
                throw std::out_of_range(
                    "Dual-layer tile-source schedule has an invalid source.");
            if (segment_source_ids.empty() || source != previous_source) {
                segment_source_ids.push_back(source);
                segment_edge_offsets.push_back(
                    static_cast<int>(source_edges.size()));
                previous_source = source;
            }
            source_edges.push_back(local_edge);
        }
        tile_segment_offsets.push_back(
            static_cast<int>(segment_source_ids.size()));
    }
    if (global_edge != neigh_indices.size()
        || source_edges.size() != neigh_indices.size())
        throw std::runtime_error(
            "Dual-layer tile-source schedule does not cover every edge exactly once.");
    segment_edge_offsets.push_back(static_cast<int>(source_edges.size()));

    const auto make_int_view = [&] (
        const std::string& label, const std::vector<int>& values) {
        auto result = Kokkos::View<int*>(
            Kokkos::view_alloc(label, Kokkos::WithoutInitializing),
            values.size());
        if (!values.empty()) {
            auto host = Kokkos::create_mirror_view(result);
            std::copy(values.begin(), values.end(), host.data());
            Kokkos::deep_copy(factorized_execution_space, result, host);
        }
        return result;
    };
    dual_layer_tile_segment_offsets = make_int_view(
        "Dual-layer tile segment offsets", tile_segment_offsets);
    dual_layer_tile_segment_offsets_host = tile_segment_offsets;
    dual_layer_segment_source_ids = make_int_view(
        "Dual-layer segment source IDs", segment_source_ids);
    dual_layer_segment_edge_offsets = make_int_view(
        "Dual-layer segment edge offsets", segment_edge_offsets);
    dual_layer_source_edges = make_int_view(
        "Dual-layer source edges", source_edges);
    dual_layer_source_segments = static_cast<int>(segment_source_ids.size());
    dual_layer_source_schedule_bytes = sizeof(int)*(
        dual_layer_tile_segment_offsets.size()
        +dual_layer_segment_source_ids.size()
        +dual_layer_segment_edge_offsets.size()
        +dual_layer_source_edges.size());
    factorized_schedule_bytes += dual_layer_source_schedule_bytes;
    dual_layer_tiled_estimate = estimate_dual_layer_tiled_graph_bytes(
        num_receivers, num_feature_nodes, neigh_indices.size(), capacity,
        static_cast<std::size_t>(dual_layer_workspace_active_edge_capacity),
        segment_source_ids.size());
    low_memory_selected_estimate = dual_layer_tiled_estimate;
    execution_planned_bytes = dual_layer_tiled_estimate;
    refresh_execution_plan_report();
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_dual_layer_tile_source_schedule_device(
    const int num_receivers,
    const int num_feature_nodes,
    const int num_edges)
{
    if (!dual_layer_tiled_plan_active)
        return;
    if (num_receivers <= 0 || num_feature_nodes < num_receivers
        || num_edges < 0
        || execution_prepared_num_neigh.extent_int(0) != num_receivers
        || execution_prepared_neigh_indices.extent_int(0) != num_edges)
        throw std::invalid_argument(
            "Dual-layer device tile-source schedule requires a valid distributed graph.");
    const int capacity = dual_layer_workspace_active_capacity;
    if (capacity <= 0)
        throw std::logic_error(
            "Dual-layer device tile-source schedule has no receiver capacity.");
    const int num_tiles = num_receivers/capacity
        +(num_receivers%capacity != 0);

    if (execution_receiver_offsets_host.size()
            != static_cast<std::size_t>(num_receivers))
        throw std::logic_error(
            "Dual-layer device tile-source schedule has no receiver offsets.");
    const int scratch_capacity = dual_layer_workspace_active_edge_capacity;
    if (scratch_capacity < 0 || scratch_capacity > num_edges)
        throw std::logic_error(
            "Dual-layer device tile-source scratch capacity is invalid.");
    auto entries_storage = Kokkos::View<DualLayerSourceEdge*>(
        Kokkos::view_alloc(
            "Dual-layer device tile-source sort entries",
            Kokkos::WithoutInitializing),
        static_cast<std::size_t>(scratch_capacity));
    dual_layer_source_edges = Kokkos::View<int*>(
        Kokkos::view_alloc(
            "Dual-layer source edges", Kokkos::WithoutInitializing),
        static_cast<std::size_t>(num_edges));

    const auto edge_sources = execution_prepared_neigh_indices;
    const auto source_edges = dual_layer_source_edges;
    auto tile_segment_offsets = std::vector<int>(
        static_cast<std::size_t>(num_tiles)+1, 0);
    for (int tile=0; tile<num_tiles; ++tile) {
        const int receiver_begin = tile*capacity;
        const int receiver_end = receiver_begin
            +std::min(capacity, num_receivers-receiver_begin);
        const int tile_edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int tile_edge_end = receiver_end == num_receivers
            ? num_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        const int tile_edges = tile_edge_end-tile_edge_begin;
        if (tile_edges < 0 || tile_edges > scratch_capacity)
            throw std::logic_error(
                "Dual-layer device tile exceeds its schedule scratch capacity.");
        if (tile_edges == 0) {
            tile_segment_offsets[static_cast<std::size_t>(tile)+1] =
                tile_segment_offsets[static_cast<std::size_t>(tile)];
            continue;
        }

        auto entries = Kokkos::subview(
            entries_storage,
            Kokkos::make_pair(
                std::size_t(0), static_cast<std::size_t>(tile_edges)));
        Kokkos::parallel_for(
            "MACEKokkos::fill_dual_layer_tile_source_sort_entries",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, tile_edges),
            KOKKOS_LAMBDA (const int local_edge) {
                entries(local_edge) = {
                    edge_sources(tile_edge_begin+local_edge), local_edge};
            });
        int invalid_sources = 0;
        Kokkos::parallel_reduce(
            "MACEKokkos::validate_dual_layer_tile_source_entries",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, tile_edges),
            KOKKOS_LAMBDA (const int local_edge, int& invalid) {
                invalid += entries(local_edge).source < 0
                    || entries(local_edge).source >= num_feature_nodes;
            },
            invalid_sources);
        if (invalid_sources != 0)
            throw std::out_of_range(
                "Dual-layer device tile-source schedule has an invalid source.");

        Kokkos::sort(
            factorized_execution_space, entries, DualLayerSourceEdgeLess{});
        Kokkos::parallel_for(
            "MACEKokkos::store_dual_layer_tile_source_edges",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, tile_edges),
            KOKKOS_LAMBDA (const int ordered_edge) {
                source_edges(tile_edge_begin+ordered_edge) =
                    entries(ordered_edge).local_edge;
            });
        int tile_segments = 0;
        Kokkos::parallel_reduce(
            "MACEKokkos::count_dual_layer_tile_source_segments",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, tile_edges),
            KOKKOS_LAMBDA (const int ordered_edge, int& count) {
                count += ordered_edge == 0
                    || entries(ordered_edge).source
                        != entries(ordered_edge-1).source;
            },
            tile_segments);
        const int previous_segments = tile_segment_offsets[
            static_cast<std::size_t>(tile)];
        if (tile_segments < 0
            || previous_segments > std::numeric_limits<int>::max()-tile_segments)
            throw std::overflow_error(
                "Dual-layer tile-source segment count exceeds 32-bit indexing.");
        tile_segment_offsets[static_cast<std::size_t>(tile)+1] =
            previous_segments+tile_segments;
    }

    const int source_segments = tile_segment_offsets.back();
    dual_layer_segment_source_ids = {};
    dual_layer_segment_edge_offsets = {};
    dual_layer_tile_segment_offsets = Kokkos::View<int*>(
        Kokkos::view_alloc(
            "Dual-layer tile segment offsets", Kokkos::WithoutInitializing),
        static_cast<std::size_t>(num_tiles)+1);
    dual_layer_segment_source_ids = Kokkos::View<int*>(
        Kokkos::view_alloc(
            "Dual-layer segment source IDs", Kokkos::WithoutInitializing),
        static_cast<std::size_t>(source_segments));
    dual_layer_segment_edge_offsets = Kokkos::View<int*>(
        Kokkos::view_alloc(
            "Dual-layer segment edge offsets", Kokkos::WithoutInitializing),
        static_cast<std::size_t>(source_segments)+1);
    const auto segment_sources = dual_layer_segment_source_ids;
    const auto segment_offsets = dual_layer_segment_edge_offsets;
    for (int tile=0; tile<num_tiles; ++tile) {
        const int receiver_begin = tile*capacity;
        const int receiver_end = receiver_begin
            +std::min(capacity, num_receivers-receiver_begin);
        const int tile_edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int tile_edge_end = receiver_end == num_receivers
            ? num_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        const int tile_edges = tile_edge_end-tile_edge_begin;
        const int segment_base = tile_segment_offsets[
            static_cast<std::size_t>(tile)];
        if (tile_edges == 0)
            continue;
        Kokkos::parallel_scan(
            "MACEKokkos::fill_dual_layer_tile_source_segments",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, tile_edges),
            KOKKOS_LAMBDA (
                const int ordered_edge, int& update, const bool final) {
                const int local_edge = source_edges(
                    tile_edge_begin+ordered_edge);
                const int source = edge_sources(tile_edge_begin+local_edge);
                const bool segment_start = ordered_edge == 0
                    || source != edge_sources(
                        tile_edge_begin+source_edges(
                            tile_edge_begin+ordered_edge-1));
                if (final && segment_start) {
                    segment_sources(segment_base+update) = source;
                    segment_offsets(segment_base+update) =
                        tile_edge_begin+ordered_edge;
                }
                update += segment_start;
            });
    }
    Kokkos::deep_copy(
        factorized_execution_space,
        Kokkos::subview(segment_offsets, source_segments), num_edges);

    const auto tile_offsets_host = Kokkos::View<
        const int*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            tile_segment_offsets.data(), tile_segment_offsets.size());
    Kokkos::deep_copy(
        factorized_execution_space,
        dual_layer_tile_segment_offsets, tile_offsets_host);
    factorized_execution_space.fence(
        "Prepare bounded dual-layer tile-source schedule");
    dual_layer_tile_segment_offsets_host = std::move(tile_segment_offsets);

    dual_layer_source_segments = source_segments;
    dual_layer_source_schedule_bytes = sizeof(int)*(
        dual_layer_tile_segment_offsets.size()
        +dual_layer_segment_source_ids.size()
        +dual_layer_segment_edge_offsets.size()
        +dual_layer_source_edges.size());
    // Backend sort implementations may allocate additional temporary storage.
    dual_layer_source_schedule_preparation_explicit_scratch_bytes =
        sizeof(DualLayerSourceEdge)*entries_storage.size();
    factorized_schedule_bytes += dual_layer_source_schedule_bytes;
    dual_layer_tiled_estimate = estimate_dual_layer_tiled_graph_bytes(
        num_receivers, num_feature_nodes, static_cast<std::size_t>(num_edges),
        capacity,
        static_cast<std::size_t>(dual_layer_workspace_active_edge_capacity),
        static_cast<std::size_t>(source_segments));
    low_memory_selected_estimate = dual_layer_tiled_estimate;
    execution_planned_bytes = dual_layer_tiled_estimate;
    refresh_execution_plan_report();
}

template <typename Precision>
bool MACEKokkos<Precision>::prepare_factorized_schedule_host(
    const int num_nodes,
    const std::vector<int>& schedule_num_neigh,
    const std::vector<int>& schedule_neigh_indices)
{
    return prepare_factorized_schedule_host(
        num_nodes, num_nodes, schedule_num_neigh, schedule_neigh_indices);
}

template <typename Precision>
bool MACEKokkos<Precision>::prepare_factorized_schedule_host(
    const int num_receivers,
    const int num_feature_nodes,
    const std::vector<int>& schedule_num_neigh,
    const std::vector<int>& schedule_neigh_indices)
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers
        || schedule_num_neigh.size() != static_cast<std::size_t>(num_receivers))
        throw std::invalid_argument("Execution R1 graph extents are inconsistent.");
    validate_graph_cardinality(
        static_cast<std::size_t>(num_receivers),
        static_cast<std::size_t>(num_feature_nodes),
        schedule_neigh_indices.size());
    const int num_edges = static_cast<int>(schedule_neigh_indices.size());
    long long counted_edges = 0;
    for (int receiver=0; receiver<num_receivers; ++receiver) {
        if (schedule_num_neigh[receiver] < 0)
            throw std::invalid_argument("Execution R1 graph has a negative receiver degree.");
        counted_edges += schedule_num_neigh[receiver];
    }
    if (counted_edges != num_edges)
        throw std::invalid_argument(
            "Execution R1 receiver degrees do not sum to the edge count.");
    for (int edge=0; edge<num_edges; ++edge) {
        const int source = schedule_neigh_indices[edge];
        if (source < 0 || source >= num_feature_nodes)
            throw std::out_of_range("Execution R1 source index is out of range.");
    }

    if (!factorized_schedule_dirty
        && num_receivers == execution_schedule_num_nodes
        && num_feature_nodes == execution_schedule_num_feature_nodes
        && schedule_num_neigh == execution_schedule_num_neigh_host
        && schedule_neigh_indices == execution_schedule_neigh_indices_host)
        return false;

    if (A0_scaled && r0_supports_low_memory()
        && !single_layer_tiled_plan_active && !dual_layer_tiled_plan_active
        && standard_r0_density_state.extent(0)
            != static_cast<std::size_t>(num_receivers))
        Kokkos::realloc(standard_r0_density_state, num_receivers);

    const int selected_chunk_size = select_factorized_chunk_size(
        num_receivers, num_feature_nodes, num_edges, schedule_num_neigh);
    const bool direct_inference = factorized_direct_profile_requested()
        || (single_layer_readout
            && mace_uses_direct_execution(streamed_edges));
    constexpr bool host_execution = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    const bool host_single_layer = single_layer_readout && host_execution;
    if (direct_inference || host_single_layer)
        factorized_chunk_size = selected_chunk_size;
    else
        resize_factorized_tile_workspace(selected_chunk_size, num_feature_nodes);
    const int num_chunks = num_receivers/factorized_chunk_size
        +(num_receivers%factorized_chunk_size != 0);
    const int source_stride = num_feature_nodes+1;
    auto source_chunk_offsets = std::vector<int>();
    if (!direct_inference) {
        const std::size_t source_schedule_entries =
            static_cast<std::size_t>(num_chunks)
            *static_cast<std::size_t>(source_stride);
        if (source_schedule_entries
                >static_cast<std::size_t>(std::numeric_limits<int>::max()))
            throw std::length_error(
                "Execution factorized source schedule exceeds 32-bit indexing.");
        source_chunk_offsets.assign(source_schedule_entries, 0);
    }
    auto first_neigh = std::vector<int>(num_receivers);
    auto chunk_edge_offsets = std::vector<int>(num_chunks+1, 0);
    const bool fixed_workspace_tiled = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    auto edge_receivers = fixed_workspace_tiled
        ? std::vector<int>() : std::vector<int>(num_edges);
    int edge = 0;
    for (int receiver=0; receiver<num_receivers; ++receiver) {
        if (receiver%factorized_chunk_size == 0)
            chunk_edge_offsets[receiver/factorized_chunk_size] = edge;
        first_neigh[receiver] = edge;
        const int chunk = receiver/factorized_chunk_size;
        const std::size_t base = static_cast<std::size_t>(chunk)
            *static_cast<std::size_t>(source_stride);
        for (int offset=0; offset<schedule_num_neigh[receiver]; ++offset, ++edge) {
            const int source = schedule_neigh_indices[edge];
            if (!direct_inference)
                source_chunk_offsets[base+source+1] += 1;
            if (!fixed_workspace_tiled)
                edge_receivers[edge] = receiver;
        }
    }
    chunk_edge_offsets[num_chunks] = edge;

    auto source_edges = std::vector<int>();
    if (!direct_inference) {
        int scheduled_edges = 0;
        for (int chunk=0; chunk<num_chunks; ++chunk) {
            const std::size_t base = static_cast<std::size_t>(chunk)
                *static_cast<std::size_t>(source_stride);
            source_chunk_offsets[base] = scheduled_edges;
            for (int source=0; source<num_feature_nodes; ++source)
                source_chunk_offsets[base+source+1] +=
                    source_chunk_offsets[base+source];
            scheduled_edges = source_chunk_offsets[base+num_feature_nodes];
        }
        if (scheduled_edges != num_edges)
            throw std::runtime_error("Execution R1 source schedule is incomplete.");

        auto source_cursor = source_chunk_offsets;
        source_edges.resize(num_edges);
        for (int original_edge=0; original_edge<num_edges; ++original_edge) {
            const int receiver = edge_receivers[original_edge];
            const int source = schedule_neigh_indices[original_edge];
            const std::size_t base =
                static_cast<std::size_t>(receiver/factorized_chunk_size)
                *static_cast<std::size_t>(source_stride);
            source_edges[source_cursor[base+source]++] = original_edge;
        }
    }

    auto direct_source_offsets = std::vector<int>();
    auto direct_source_edges = std::vector<int>();
    if (!single_layer_readout && !dual_layer_tiled_plan_active) {
        direct_source_offsets.assign(num_feature_nodes+1, 0);
        for (const int source : schedule_neigh_indices)
            direct_source_offsets[source+1] += 1;
        for (int source=0; source<num_feature_nodes; ++source)
            direct_source_offsets[source+1] += direct_source_offsets[source];
        auto direct_source_cursor = direct_source_offsets;
        direct_source_edges.resize(num_edges);
        for (int original_edge=0; original_edge<num_edges; ++original_edge) {
            const int source = schedule_neigh_indices[original_edge];
            direct_source_edges[direct_source_cursor[source]++] = original_edge;
        }
    }

    reserve_factorized_graph_device_capacity(
        num_receivers, num_feature_nodes, num_edges,
        source_chunk_offsets.size(), source_edges.size());
    copy_int_prefix(
        factorized_execution_space, execution_source_chunk_offsets_storage,
        execution_source_chunk_offsets, source_chunk_offsets);
    copy_int_prefix(
        factorized_execution_space, execution_source_edges_storage,
        execution_source_edges, source_edges);
    copy_int_prefix(
        factorized_execution_space, execution_direct_source_offsets_storage,
        execution_direct_source_offsets, direct_source_offsets);
    copy_int_prefix(
        factorized_execution_space, execution_direct_source_edges_storage,
        execution_direct_source_edges, direct_source_edges);
    if (!fixed_workspace_tiled)
        copy_int_prefix(
            factorized_execution_space, execution_edge_receivers_storage,
            execution_edge_receivers, edge_receivers);
    if (fixed_workspace_tiled)
        streamed_first_neigh = {};
    else
        copy_int_prefix(
            factorized_execution_space, streamed_first_neigh_storage,
            streamed_first_neigh, first_neigh);
    execution_schedule_num_neigh_host = schedule_num_neigh;
    execution_schedule_neigh_indices_host = schedule_neigh_indices;
    execution_receiver_offsets_host = std::move(first_neigh);
    execution_chunk_edge_offsets_host = std::move(chunk_edge_offsets);
    execution_schedule_num_nodes = num_receivers;
    execution_schedule_num_feature_nodes = num_feature_nodes;
    execution_schedule_num_chunks = num_chunks;
    int maximum_chunk_edges = 0;
    for (int chunk=0; chunk<num_chunks; ++chunk)
        maximum_chunk_edges = std::max(
            maximum_chunk_edges,
            execution_chunk_edge_offsets_host[chunk+1]
                -execution_chunk_edge_offsets_host[chunk]);
    if (!direct_inference) {
        if (execution_radial_values.extent(0)
                != static_cast<std::size_t>(maximum_chunk_edges)
            || execution_radial_values.extent(1)
                != static_cast<std::size_t>(factorized_embedding_width))
            Kokkos::realloc(
                execution_radial_values,
                maximum_chunk_edges, factorized_embedding_width);
        if (execution_radial_derivatives.extent(0)
                != static_cast<std::size_t>(maximum_chunk_edges)
            || execution_radial_derivatives.extent(1)
                != static_cast<std::size_t>(factorized_embedding_width))
            Kokkos::realloc(
                execution_radial_derivatives,
                maximum_chunk_edges,
                factorized_embedding_width);
    }
    factorized_coupling_capacity_bytes = sizeof(Precision)
        *static_cast<std::size_t>(maximum_chunk_edges)
        *static_cast<std::size_t>(factorized_max_coupling_columns);
    factorized_schedule_build_count += 1;
    factorized_schedule_bytes = sizeof(int)*(
        execution_source_chunk_offsets.size()+execution_source_edges.size()
        +execution_direct_source_offsets.size()+execution_direct_source_edges.size()
        +execution_edge_receivers.size()+streamed_first_neigh.size());
    update_factorized_workspace_accounting();
    plan_factorized_reverse_cache();
    if (use_factorized_direct_inference() || host_single_layer)
        release_factorized_stateful_workspace();
    factorized_schedule_dirty = false;
    return true;
}

template <typename Precision>
bool MACEKokkos<Precision>::reserve_factorized_graph_device_capacity(
    const int num_receivers,
    const int num_feature_nodes,
    const int num_edges,
    const std::size_t source_chunk_offset_count,
    const std::size_t source_edge_count)
{
    if (num_receivers < 0 || num_feature_nodes < num_receivers || num_edges < 0)
        throw std::invalid_argument(
            "Factorized graph capacity requires valid active cardinalities.");
    const std::size_t receiver_capacity = static_cast<std::size_t>(std::max(
        num_receivers, execution_planned_receivers));
    const std::size_t feature_capacity = static_cast<std::size_t>(std::max(
        num_feature_nodes, execution_planned_feature_nodes));
    const std::size_t edge_capacity = static_cast<std::size_t>(std::max(
        num_edges, execution_planned_edges));
    const std::size_t direct_offset_capacity =
        (single_layer_readout || dual_layer_tiled_plan_active)
        ? 0 : feature_capacity+1;
    const std::size_t direct_edge_capacity =
        (single_layer_readout || dual_layer_tiled_plan_active)
        ? 0 : edge_capacity;
    const std::size_t source_offset_capacity = source_chunk_offset_count;
    const std::size_t source_schedule_edge_capacity = source_edge_count;
    const bool fixed_workspace_tiled = single_layer_tiled_plan_active
        || dual_layer_tiled_plan_active;
    const std::size_t receiver_offset_capacity = fixed_workspace_tiled
        ? 0 : receiver_capacity;
    const std::size_t edge_receiver_capacity = fixed_workspace_tiled
        ? 0 : edge_capacity;
    const std::size_t neighbor_type_capacity = fixed_workspace_tiled
        ? 0 : edge_capacity;

    const bool fits =
        execution_prepared_receiver_feature_indices_storage.extent(0)
            >= receiver_capacity
        && execution_prepared_node_types_storage.extent(0) >= receiver_capacity
        && execution_prepared_feature_types_storage.extent(0) >= feature_capacity
        && execution_prepared_num_neigh_storage.extent(0) >= receiver_capacity
        && execution_prepared_neigh_indices_storage.extent(0) >= edge_capacity
        && (neighbor_type_capacity == 0
            ? execution_prepared_neigh_types_storage.extent(0) == 0
            : execution_prepared_neigh_types_storage.extent(0)
                >= neighbor_type_capacity)
        && streamed_first_neigh_storage.extent(0) >= receiver_offset_capacity
        && execution_source_chunk_offsets_storage.extent(0)
            >= source_offset_capacity
        && execution_source_edges_storage.extent(0)
            >= source_schedule_edge_capacity
        && execution_direct_source_offsets_storage.extent(0)
            >= direct_offset_capacity
        && execution_direct_source_edges_storage.extent(0)
            >= direct_edge_capacity
        && (edge_receiver_capacity == 0
            ? execution_edge_receivers_storage.extent(0) == 0
            : execution_edge_receivers_storage.extent(0)
                >= edge_receiver_capacity);
    if (fits)
        return false;

    factorized_execution_space.fence(
        "Replace factorized graph device capacity bundle");
    execution_source_chunk_offsets = {};
    execution_source_edges = {};
    execution_direct_source_offsets = {};
    execution_direct_source_edges = {};
    execution_edge_receivers = {};
    execution_prepared_receiver_feature_indices = {};
    execution_prepared_node_types = {};
    execution_prepared_feature_types = {};
    execution_prepared_num_neigh = {};
    execution_prepared_neigh_indices = {};
    execution_prepared_neigh_types = {};
    streamed_first_neigh = {};
    execution_source_chunk_offsets_storage = {};
    execution_source_edges_storage = {};
    execution_direct_source_offsets_storage = {};
    execution_direct_source_edges_storage = {};
    execution_edge_receivers_storage = {};
    execution_prepared_receiver_feature_indices_storage = {};
    execution_prepared_node_types_storage = {};
    execution_prepared_feature_types_storage = {};
    execution_prepared_num_neigh_storage = {};
    execution_prepared_neigh_indices_storage = {};
    execution_prepared_neigh_types_storage = {};
    streamed_first_neigh_storage = {};

    const auto allocate = [] (const char* label, const std::size_t count) {
        return count == 0 ? Kokkos::View<int*>() : Kokkos::View<int*>(
            Kokkos::view_alloc(
                std::string(label), Kokkos::WithoutInitializing), count);
    };
    execution_source_chunk_offsets_storage = allocate(
        "Execution R1 source chunk offsets capacity", source_offset_capacity);
    execution_source_edges_storage = allocate(
        "Execution R1 source edges capacity", source_schedule_edge_capacity);
    execution_direct_source_offsets_storage = allocate(
        "Execution direct R1 source offsets capacity", direct_offset_capacity);
    execution_direct_source_edges_storage = allocate(
        "Execution direct R1 source edges capacity", direct_edge_capacity);
    execution_edge_receivers_storage = allocate(
        "Execution R1 edge receivers capacity", edge_receiver_capacity);
    execution_prepared_receiver_feature_indices_storage = allocate(
        "Execution R1 receiver feature indices capacity", receiver_capacity);
    execution_prepared_node_types_storage = allocate(
        "Execution R1 prepared node types capacity", receiver_capacity);
    execution_prepared_feature_types_storage = allocate(
        "Execution R1 prepared feature types capacity", feature_capacity);
    execution_prepared_num_neigh_storage = allocate(
        "Execution R1 prepared receiver degrees capacity", receiver_capacity);
    execution_prepared_neigh_indices_storage = allocate(
        "Execution R1 prepared source indices capacity", edge_capacity);
    execution_prepared_neigh_types_storage = allocate(
        "Execution R1 prepared neighbor types capacity", neighbor_type_capacity);
    streamed_first_neigh_storage = allocate(
        "Execution R1 receiver offsets capacity", receiver_offset_capacity);
    factorized_graph_device_capacity_bytes = sizeof(int)*(
        source_offset_capacity+source_schedule_edge_capacity
        +direct_offset_capacity+direct_edge_capacity
        +edge_capacity+edge_receiver_capacity+neighbor_type_capacity
        +3*receiver_capacity+feature_capacity+receiver_offset_capacity);
    factorized_graph_device_replacement_count += 1;
    return true;
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_active_types(std::vector<int> node_types)
{
    if (!uses_compact_radial)
        return;

    std::sort(node_types.begin(), node_types.end());
    node_types.erase(std::unique(node_types.begin(), node_types.end()), node_types.end());
    if (node_types.empty())
        throw std::invalid_argument("MACEKokkos compact radial cache requires at least one active type.");
    for (const int type : node_types)
        if (type < 0 || type >= atomic_numbers_host.size())
            throw std::out_of_range("MACEKokkos active type index is out of range.");
    if (node_types == active_types
        && (!mace_uses_prepared_execution(streamed_edges) || factorized_ready))
        return;

    Kokkos::fence();
    const int active_count = node_types.size();
    const int pair_count = active_count*(active_count+1)/2;
    auto pair_tables = std::vector<CompactRadialPairTables>();
    pair_tables.reserve(pair_count);
    auto factorized_pair_tables = std::vector<CompactRadialFactorizedPairTables>();
    if (mace_uses_prepared_execution(streamed_edges) && !single_layer_readout)
        factorized_pair_tables.reserve(pair_count);
    for (int active_i=0; active_i<active_count; ++active_i)
        for (int active_j=active_i; active_j<active_count; ++active_j) {
            pair_tables.push_back(compact_radial_model->materialize_pair(
                node_types[active_i], node_types[active_j]));
            if (mace_uses_prepared_execution(streamed_edges)
                && !single_layer_readout)
                factorized_pair_tables.push_back(
                    compact_radial_model->materialize_factorized_pair(
                        node_types[active_i], node_types[active_j]));
        }

    const double h = compact_radial_model->spline_h();
    const double x0 = compact_radial_model->spline_min();
    const int spline_nodes = compact_radial_model->num_spline_points();
    const int R0_functions = (l_max+1)*num_channels;
    auto new_R0_coefficients = Kokkos::View<Precision****,Kokkos::LayoutRight>(
        "R0 compact coefficients", active_count*active_count,
        spline_nodes-1, 4, R0_functions);
    auto h_R0_coefficients = Kokkos::create_mirror_view(new_R0_coefficients);
    for (int active_i=0; active_i<active_count; ++active_i) {
        for (int active_j=0; active_j<active_count; ++active_j) {
            const int ordered_pair = active_i*active_count+active_j;
            const int unordered_pair = (active_i <= active_j)
                ? active_i*(2*active_count-active_i-1)/2+active_j
                : active_j*(2*active_count-active_j-1)/2+active_i;
            const int global_i = node_types[active_i];
            const int global_j = node_types[active_j];
            const auto& values = pair_tables[unordered_pair].R0.values;
            const auto& derivatives = pair_tables[unordered_pair].R0.derivatives;
            if (values.size() != R0_functions || derivatives.size() != R0_functions)
                throw std::runtime_error("Compact radial R0 output has an invalid size.");
            for (int node=0; node<spline_nodes-1; ++node) {
                for (int lk=0; lk<R0_functions; ++lk) {
                    const double value = values[lk][node];
                    const double deriv = derivatives[lk][node];
                    const double next_value = values[lk][node+1];
                    const double next_deriv = derivatives[lk][node+1];
                    h_R0_coefficients(ordered_pair,node,0,lk) = static_cast<Precision>(value);
                    h_R0_coefficients(ordered_pair,node,1,lk) = static_cast<Precision>(deriv);
                    h_R0_coefficients(ordered_pair,node,2,lk) = static_cast<Precision>(
                        (-3*value-2*h*deriv+3*next_value-h*next_deriv)/(h*h));
                    h_R0_coefficients(ordered_pair,node,3,lk) = static_cast<Precision>(
                        (2*value+h*deriv-2*next_value+h*next_deriv)/(h*h*h));
                }

                for (int lk=0; lk<R0_functions; ++lk) {
                    const int channel = lk%num_channels;
                    const auto H0_weight = static_cast<Precision>(
                        H0_weights_host[global_j*num_channels+channel]);
                    for (int coefficient=0; coefficient<4; ++coefficient)
                        h_R0_coefficients(ordered_pair,node,coefficient,lk) *= H0_weight;
                }

                for (int l=0; l<=l_max; ++l) {
                    auto original = std::vector<Precision>(4*num_channels);
                    for (int coefficient=0; coefficient<4; ++coefficient)
                        for (int channel=0; channel<num_channels; ++channel)
                            original[coefficient*num_channels+channel] =
                                h_R0_coefficients(
                                    ordered_pair,node,coefficient,l*num_channels+channel);
                    for (int coefficient=0; coefficient<4; ++coefficient) {
                        for (int channel=0; channel<num_channels; ++channel) {
                            Precision fused = 0.0;
                            for (int input_channel=0; input_channel<num_channels; ++input_channel)
                                fused += static_cast<Precision>(
                                    A0_weights_host[global_i][l][input_channel*num_channels+channel])
                                    * original[coefficient*num_channels+input_channel];
                            h_R0_coefficients(
                                ordered_pair,node,coefficient,l*num_channels+channel) = fused;
                        }
                    }
                }
            }
        }
    }
    Kokkos::deep_copy(new_R0_coefficients, h_R0_coefficients);

    auto R1_values = std::vector<std::vector<std::vector<double>>>();
    auto R1_derivatives = std::vector<std::vector<std::vector<double>>>();
    auto A0_values = std::vector<std::vector<std::vector<double>>>();
    auto A0_derivatives = std::vector<std::vector<std::vector<double>>>();
    auto A1_values = std::vector<std::vector<std::vector<double>>>();
    auto A1_derivatives = std::vector<std::vector<std::vector<double>>>();
    if (!single_layer_readout) {
        R1_values.reserve(pair_count);
        R1_derivatives.reserve(pair_count);
    }
    for (auto& tables : pair_tables) {
        if (!single_layer_readout) {
            const auto expected_R1 = static_cast<std::size_t>(
                Phi1_l.size()*num_channels);
            if (tables.R1.values.size() != expected_R1
                || tables.R1.derivatives.size() != expected_R1)
                throw std::runtime_error(
                    "Compact radial R1 output has an invalid size.");
            R1_values.push_back(std::move(tables.R1.values));
            R1_derivatives.push_back(std::move(tables.R1.derivatives));
        }
        if (A0_scaled) {
            A0_values.push_back(std::move(tables.A0.values));
            A0_derivatives.push_back(std::move(tables.A0.derivatives));
        }
        if (A1_scaled) {
            A1_values.push_back(std::move(tables.A1.values));
            A1_derivatives.push_back(std::move(tables.A1.derivatives));
        }
    }
    auto new_radial_1 = RadialFunctionSetKokkos<Precision>();
    if (!single_layer_readout)
        new_radial_1 = RadialFunctionSetKokkos<Precision>(
            h, R1_values, R1_derivatives, x0, r_cut);
    auto new_A0_splines = RadialFunctionSetKokkos<double>();
    auto new_A1_splines = RadialFunctionSetKokkos<double>();
    if (A0_scaled)
        new_A0_splines =
            RadialFunctionSetKokkos<double>(h, A0_values, A0_derivatives, x0);
    if (A1_scaled)
        new_A1_splines =
            RadialFunctionSetKokkos<double>(h, A1_values, A1_derivatives, x0);

    auto new_type_to_active = Kokkos::View<int*>("compact type_to_active", atomic_numbers_host.size());
    auto h_type_to_active = Kokkos::create_mirror_view(new_type_to_active);
    Kokkos::deep_copy(h_type_to_active, -1);
    auto new_active_atomic_numbers = std::vector<int>();
    new_active_atomic_numbers.reserve(active_count);
    for (int active=0; active<active_count; ++active) {
        h_type_to_active(node_types[active]) = active;
        new_active_atomic_numbers.push_back(atomic_numbers_host[node_types[active]]);
    }
    Kokkos::deep_copy(new_type_to_active, h_type_to_active);
    Kokkos::fence();

    R0_spline_h = h;
    R0_spline_min = x0;
    R0_spline_coefficients = new_R0_coefficients;
    radial_1 = std::move(new_radial_1);
    if (A0_scaled)
        A0_splines = std::move(new_A0_splines);
    if (A1_scaled)
        A1_splines = std::move(new_A1_splines);
    if (mace_uses_prepared_execution(streamed_edges)
        && !single_layer_readout)
        prepare_factorized_model(factorized_pair_tables, h, x0);
    else if (mace_uses_prepared_execution(streamed_edges))
        factorized_ready = true;
    type_to_active = new_type_to_active;
    num_active_types = active_count;
    active_atomic_numbers = std::move(new_active_atomic_numbers);
    active_types = std::move(node_types);
}

template <typename Precision>
void MACEKokkos<Precision>::prepare_factorized_model(
    const std::vector<CompactRadialFactorizedPairTables>& pair_tables,
    double h,
    double x0)
{
    if (pair_tables.empty())
        throw std::invalid_argument("Execution R1 requires at least one active species pair.");
    release_factorized_coupling_workspace();
    factorized_chunk_size = factorized_default_chunk_size;
    factorized_schedule_dirty = true;
    invalidate_factorized_prepared_graph();

    const auto& reference = pair_tables.front().R1;
    const int embedding_width = reference.embedding_width;
    const int num_paths = static_cast<int>(Phi1_l.extent(0));
    const int expected_output_width = num_paths*num_channels;
    if (embedding_width <= 0 || reference.output_width != expected_output_width)
        throw std::runtime_error("Execution R1 radial projection has incompatible extents.");

    auto penultimate_values = std::vector<std::vector<std::vector<double>>>();
    auto penultimate_derivatives = std::vector<std::vector<std::vector<double>>>();
    penultimate_values.reserve(pair_tables.size());
    penultimate_derivatives.reserve(pair_tables.size());
    for (const auto& pair : pair_tables) {
        const auto& factorized = pair.R1;
        if (factorized.embedding_width != embedding_width
            || factorized.output_width != expected_output_width
            || factorized.final_projection.size() != reference.final_projection.size())
            throw std::runtime_error("Execution R1 species-pair factorizations disagree.");
        for (std::size_t index=0; index<reference.final_projection.size(); ++index)
            if (factorized.final_projection[index] != reference.final_projection[index])
                throw std::runtime_error(
                    "Execution R1 final projection must be shared across species pairs.");
        penultimate_values.push_back(factorized.penultimate.values);
        penultimate_derivatives.push_back(factorized.penultimate.derivatives);
    }
    auto new_radial = RadialFunctionSetKokkos<Precision>(
        h, std::move(penultimate_values), std::move(penultimate_derivatives),
        x0, r_cut);

    const auto h_path_l = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l);
    const auto h_path_l1 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l1);
    const auto h_path_l2 = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_l2);
    const auto h_lme = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_lme);
    const auto h_rows =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_lelm1lm2);
    const auto h_coefficients =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_clebsch_gordan);
    const auto h_lm1_rows =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_lm1);
    const auto h_lm2_rows =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Phi1_lm2);

    auto group_path_offsets = std::vector<int>(l_max+2, 0);
    for (int path=0; path<num_paths; ++path) {
        const int l = h_path_l(path);
        if (l < 0 || l > l_max)
            throw std::runtime_error("Execution R1 path has an invalid output irrep.");
        group_path_offsets[l+1] += 1;
    }
    for (int l=0; l<=l_max; ++l)
        group_path_offsets[l+1] += group_path_offsets[l];
    group_path_offsets[l_max+1] = num_paths;
    auto group_paths = std::vector<int>(num_paths);
    auto group_cursor = group_path_offsets;
    for (int path=0; path<num_paths; ++path)
        group_paths[group_cursor[h_path_l(path)]++] = path;

    auto path_lme_offsets = std::vector<int>(num_paths+1, 0);
    for (int path=0; path<num_paths; ++path)
        path_lme_offsets[path+1] = path_lme_offsets[path]+2*h_path_l(path)+1;
    if (path_lme_offsets.back() != num_lme)
        throw std::runtime_error("Execution R1 path/component layout is inconsistent.");

    auto path_component_lme = std::vector<int>(num_lme, -1);
    auto lme_component = std::vector<int>(num_lme, -1);
    auto group_lme_offsets = std::vector<int>(l_max+2, 0);
    int group_lme_begin = 0;
    for (int l=0; l<=l_max; ++l) {
        const int num_eta = group_path_offsets[l+1]-group_path_offsets[l];
        for (int eta=0; eta<num_eta; ++eta) {
            const int path = group_paths[group_path_offsets[l]+eta];
            for (int component=0; component<2*l+1; ++component) {
                const int lme = group_lme_begin+component*num_eta+eta;
                path_component_lme[path_lme_offsets[path]+component] = lme;
                lme_component[lme] = component;
            }
        }
        group_lme_begin += (2*l+1)*num_eta;
        group_lme_offsets[l+1] = group_lme_begin;
    }
    if (group_lme_begin != num_lme
        || std::find(path_component_lme.begin(), path_component_lme.end(), -1)
            != path_component_lme.end())
        throw std::runtime_error("Execution R1 packed lme mapping is incomplete.");

    auto cg_offsets = std::vector<int>(num_lme+1, 0);
    for (std::size_t term=0; term<h_lme.extent(0); ++term) {
        const int lme = h_lme(term);
        if (lme < 0 || lme >= num_lme)
            throw std::runtime_error("Execution R1 CG output index is invalid.");
        cg_offsets[lme+1] += 1;
    }
    for (int lme=0; lme<num_lme; ++lme)
        cg_offsets[lme+1] += cg_offsets[lme];
    auto cg_rows = std::vector<int>(h_lme.extent(0));
    auto cg_coefficients = std::vector<Precision>(h_lme.extent(0));
    auto cg_cursor = cg_offsets;
    for (std::size_t term=0; term<h_lme.extent(0); ++term) {
        const int destination = cg_cursor[h_lme(term)]++;
        cg_rows[destination] = h_rows(term);
        cg_coefficients[destination] = h_coefficients(term);
    }

    const int source_key_count = (l_max+1)*num_LM;
    auto source_lm_offsets = std::vector<int>(source_key_count+1, 0);
    for (int l=0; l<=l_max; ++l) {
        const int components = 2*l+1;
        const int group_begin = group_path_offsets[l];
        const int num_eta = group_path_offsets[l+1]-group_begin;
        for (int eta=0; eta<num_eta; ++eta) {
            const int path = group_paths[group_begin+eta];
            for (int component=0; component<components; ++component) {
                const int lme = path_component_lme[
                    path_lme_offsets[path]+component];
                for (int term=cg_offsets[lme]; term<cg_offsets[lme+1]; ++term) {
                    const int lm2 = h_lm2_rows(cg_rows[term]);
                    if (lm2 < 0 || lm2 >= num_LM)
                        throw std::runtime_error(
                            "Execution R1 source harmonic index is invalid.");
                    source_lm_offsets[l*num_LM+lm2+1] += 1;
                }
            }
        }
    }
    for (int key=0; key<source_key_count; ++key)
        source_lm_offsets[key+1] += source_lm_offsets[key];
    auto source_eta_components = std::vector<int>(h_lme.extent(0));
    auto source_lm1 = std::vector<int>(h_lme.extent(0));
    auto source_coefficients = std::vector<Precision>(h_lme.extent(0));
    auto source_cursor = source_lm_offsets;
    for (int l=0; l<=l_max; ++l) {
        const int components = 2*l+1;
        const int group_begin = group_path_offsets[l];
        const int num_eta = group_path_offsets[l+1]-group_begin;
        for (int eta=0; eta<num_eta; ++eta) {
            const int path = group_paths[group_begin+eta];
            for (int component=0; component<components; ++component) {
                const int lme = path_component_lme[
                    path_lme_offsets[path]+component];
                for (int term=cg_offsets[lme]; term<cg_offsets[lme+1]; ++term) {
                    const int row = cg_rows[term];
                    const int lm2 = h_lm2_rows(row);
                    const int destination = source_cursor[l*num_LM+lm2]++;
                    source_eta_components[destination] =
                        eta*components+component;
                    source_lm1[destination] = h_lm1_rows(row);
                    source_coefficients[destination] = cg_coefficients[term];
                }
            }
        }
    }

    auto row_cg_offsets = std::vector<int>(num_lelm1lm2+1, 0);
    for (std::size_t term=0; term<h_rows.extent(0); ++term) {
        const int row = h_rows(term);
        if (row < 0 || row >= num_lelm1lm2)
            throw std::runtime_error("Execution R1 CG raw-row index is invalid.");
        row_cg_offsets[row+1] += 1;
    }
    for (int row=0; row<num_lelm1lm2; ++row)
        row_cg_offsets[row+1] += row_cg_offsets[row];
    auto row_cg_lme = std::vector<int>(h_rows.extent(0));
    auto row_cg_coefficients = std::vector<Precision>(h_rows.extent(0));
    auto row_cg_cursor = row_cg_offsets;
    for (std::size_t term=0; term<h_rows.extent(0); ++term) {
        const int destination = row_cg_cursor[h_rows(term)]++;
        row_cg_lme[destination] = h_lme(term);
        row_cg_coefficients[destination] = h_coefficients(term);
    }

    using Matrix = Kokkos::View<Precision**,Kokkos::LayoutRight>;
    using MatrixArray = Kokkos::View<Matrix*,Kokkos::SharedSpace>;
    auto projection = MatrixArray(
        Kokkos::view_alloc("Execution R1 projections", Kokkos::SequentialHostInit),
        l_max+1);
    auto projection_trans = MatrixArray(
        Kokkos::view_alloc(
            "Execution R1 transposed projections", Kokkos::SequentialHostInit),
        l_max+1);
    auto final_projection = MatrixArray(
        Kokkos::view_alloc(
            "Execution R1 final radial projections", Kokkos::SequentialHostInit),
        l_max+1);
    std::size_t receiver_projection_elements = 0;
    for (int l=0; l<=l_max; ++l) {
        const int num_eta = group_path_offsets[l+1]-group_path_offsets[l];
        receiver_projection_elements += static_cast<std::size_t>(
            num_eta*embedding_width*num_channels)*num_channels;
    }
    auto receiver_projection = Kokkos::View<Precision*>(
        Kokkos::view_alloc(
            "Receiver-factorized R1 projection",
            Kokkos::WithoutInitializing),
        receiver_projection_elements);
    auto h_receiver_projection = Kokkos::create_mirror_view(receiver_projection);
    std::size_t receiver_projection_offset = 0;
    int maximum_coupling_columns = 0;
    for (int l=0; l<=l_max; ++l) {
        const int num_eta = group_path_offsets[l+1]-group_path_offsets[l];
        maximum_coupling_columns = std::max(
            maximum_coupling_columns,
            num_eta*(2*l+1)*num_channels);
        const int contracted_width = num_eta*embedding_width*num_channels;
        projection(l) = Matrix(
            Kokkos::view_alloc(
                "Execution R1 projection " + std::to_string(l),
                Kokkos::WithoutInitializing),
            contracted_width,
            num_channels);
        projection_trans(l) = Matrix(
            Kokkos::view_alloc(
                "Execution R1 transposed projection " + std::to_string(l),
                Kokkos::WithoutInitializing),
            num_channels,
            contracted_width);
        final_projection(l) = Matrix(
            Kokkos::view_alloc(
                "Execution R1 final radial projection " + std::to_string(l),
                Kokkos::WithoutInitializing),
            num_eta*num_channels,
            embedding_width);
        auto h_projection = Kokkos::create_mirror_view(projection(l));
        auto h_projection_trans = Kokkos::create_mirror_view(projection_trans(l));
        auto h_final_projection =
            Kokkos::create_mirror_view(final_projection(l));
        const auto h_A1 =
            Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A1_weights(l));
        for (int eta=0; eta<num_eta; ++eta) {
            const int path = group_paths[group_path_offsets[l]+eta];
            for (int q=0; q<embedding_width; ++q) {
                for (int input_channel=0; input_channel<num_channels; ++input_channel) {
                    const int row = (eta*embedding_width+q)*num_channels+input_channel;
                    const Precision radial_weight = static_cast<Precision>(
                        reference.final_projection[
                            (path*num_channels+input_channel)*embedding_width+q]);
                    h_final_projection(
                        eta*num_channels+input_channel,q) = radial_weight;
                    for (int output_channel=0; output_channel<num_channels; ++output_channel) {
                        const Precision value = radial_weight
                            *h_A1(eta*num_channels+input_channel, output_channel);
                        h_projection(row,output_channel) = value;
                        h_projection_trans(output_channel,row) = value;
                        h_receiver_projection(
                            receiver_projection_offset
                                +static_cast<std::size_t>(row)*num_channels
                                +output_channel) = value;
                    }
                }
            }
        }
        Kokkos::deep_copy(projection(l), h_projection);
        Kokkos::deep_copy(projection_trans(l), h_projection_trans);
        Kokkos::deep_copy(final_projection(l), h_final_projection);
        receiver_projection_offset +=
            static_cast<std::size_t>(contracted_width)*num_channels;
    }
    Kokkos::deep_copy(receiver_projection, h_receiver_projection);

    execution_radial_1 = std::move(new_radial);
    factorized_embedding_width = embedding_width;
    execution_group_path_offsets_host = group_path_offsets;
    execution_group_lme_offsets_host = std::move(group_lme_offsets);
    execution_group_paths = toKokkosView("Execution R1 group paths", group_paths);
    execution_path_component_offsets = toKokkosView(
        "Execution R1 path component offsets", path_lme_offsets);
    execution_path_component_lme = toKokkosView(
        "Execution R1 path component lme", path_component_lme);
    execution_cg_offsets = toKokkosView("Execution R1 CG offsets", cg_offsets);
    execution_cg_rows = toKokkosView("Execution R1 CG rows", cg_rows);
    execution_cg_coefficients = toKokkosView(
        "Execution R1 CG coefficients", cg_coefficients);
    execution_source_lm_offsets = toKokkosView(
        "Execution R1 source harmonic offsets", source_lm_offsets);
    execution_source_eta_components = toKokkosView(
        "Execution R1 source eta components", source_eta_components);
    execution_source_lm1 = toKokkosView(
        "Execution R1 source lm1", source_lm1);
    execution_source_coefficients = toKokkosView(
        "Execution R1 source coefficients", source_coefficients);
    execution_row_cg_offsets = toKokkosView(
        "Execution R1 row CG offsets", row_cg_offsets);
    execution_row_cg_lme = toKokkosView("Execution R1 row CG lme", row_cg_lme);
    execution_row_cg_coefficients = toKokkosView(
        "Execution R1 row CG coefficients", row_cg_coefficients);
    execution_lme_component = toKokkosView(
        "Execution R1 lme components", lme_component);
    execution_projection = std::move(projection);
    execution_projection_trans = std::move(projection_trans);
    execution_final_projection = std::move(final_projection);
    execution_receiver_projection = std::move(receiver_projection);
    factorized_max_coupling_columns = maximum_coupling_columns;
    factorized_workspace_arena = {};
    factorized_state_workspace_bytes = 0;
    factorized_chunk_size = factorized_default_chunk_size;
    factorized_compact_workspace_bytes = 0;
    update_factorized_workspace_accounting();
    factorized_ready = true;
    Kokkos::fence();
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
