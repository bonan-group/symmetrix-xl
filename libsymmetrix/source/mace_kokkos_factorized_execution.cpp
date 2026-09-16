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
#include "mace_kokkos_factorized_blas_detail.hpp"

#include "mace_kokkos_kernel_launch_detail.hpp"

#include "mace_kokkos_jit_plugin_detail.hpp"

template <typename Precision>
void MACEKokkos<Precision>::compute_factorized(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> r)
{
    if (!factorized_ready)
        throw std::runtime_error("Execution R1 model cache has not been prepared.");
    const auto execution_space = factorized_execution_space;
    ensure_factorized_stateful_workspace();
    const bool plugin_tiled_fallback =
        factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && factorized_tiled_ready;
    if (factorized_source_strategy == FactorizedSourceStrategy::tiled_coupling
        || plugin_tiled_fallback) {
        if (!factorized_tiled_ready)
            throw std::runtime_error(
                "Execution R1 tiled coupling workspace exceeds 64 MiB.");
        ensure_factorized_coupling_workspace(
            factorized_max_coupling_columns);
    }
    if (A1.extent(0) < static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(A1, num_nodes, num_lm, num_channels);

    using TeamMember = Kokkos::TeamPolicy<>::member_type;
    using ScratchView = Kokkos::View<
        Precision*, typename TeamMember::scratch_memory_space,
        Kokkos::MemoryUnmanaged>;
    const int channels = num_channels;
    const int harmonics = num_lm;
    const int embedding = factorized_embedding_width;
    const int active_type_count = num_active_types;
    const auto first_neigh = streamed_first_neigh;
    const auto type_map = type_to_active;
    const auto radial = execution_radial_1;
    const auto group_paths = execution_group_paths;
    const auto path_component_offsets = execution_path_component_offsets;
    const auto path_component_lme = execution_path_component_lme;
    const auto cg_offsets = execution_cg_offsets;
    const auto cg_rows = execution_cg_rows;
    const auto cg_coefficients = execution_cg_coefficients;
    const auto lm1_rows = Phi1_lm1;
    const auto lm2_rows = Phi1_lm2;
    const auto harmonics_values = Y;
    const auto neighbor_features = H1;
    auto output_features = A1;
    const auto edge_receivers = execution_edge_receivers;
    auto radial_values = execution_radial_values;
    auto coupling = execution_coupling_adjoint;
    auto aggregate = factorized_packed_state_view();
    auto compact_message = factorized_compact_message_view();
    const auto final_projections = execution_final_projection;
    const auto a1_weights = A1_weights;
    const bool tiled_execution =
        factorized_source_strategy == FactorizedSourceStrategy::tiled_coupling
        || plugin_tiled_fallback;

    if (tiled_execution) {
        if (!factorized_tiled_ready)
            throw std::runtime_error(
                "Execution R1 tiled coupling workspace exceeds 64 MiB.");
        if (execution_schedule_num_nodes != num_nodes
            || static_cast<int>(edge_receivers.extent(0))
                != static_cast<int>(neigh_indices.extent(0)))
            throw std::runtime_error(
                "Execution R1 source schedule has not been prepared.");

        for (int receiver_begin=0; receiver_begin<num_nodes;
             receiver_begin += factorized_chunk_size) {
            const int receiver_count = std::min(
                factorized_chunk_size, num_nodes-receiver_begin);
            const int chunk = receiver_begin/factorized_chunk_size;
            const int chunk_edge_begin = execution_chunk_edge_offsets_host[chunk];
            const int chunk_edge_end = execution_chunk_edge_offsets_host[chunk+1];
            const int chunk_edges = chunk_edge_end-chunk_edge_begin;

            Kokkos::parallel_for(
                "MACEKokkos::cache_factorized_forward_radial",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, chunk_edges*embedding),
                KOKKOS_LAMBDA (const int flat) {
                    const int q = flat%embedding;
                    const int local_edge = flat/embedding;
                    const int edge = chunk_edge_begin+local_edge;
                    const int receiver = edge_receivers(edge);
                    const int type_i = type_map(node_types(receiver));
                    const int type_j = type_map(neigh_types(edge));
                    const int edge_type = type_i <= type_j
                        ? type_i*(2*active_type_count-type_i-1)/2+type_j
                        : type_j*(2*active_type_count-type_j-1)/2+type_i;
                    radial_values(local_edge,q) = radial.evaluate_function(
                        edge_type, radial.evaluation_point(r(edge)), q);
                });

            bool uniform_degree = receiver_count > 0;
            const int tile_degree = uniform_degree
                ? execution_schedule_num_neigh_host[receiver_begin] : 0;
            for (int local_receiver=1;
                 local_receiver<receiver_count; ++local_receiver)
                uniform_degree = uniform_degree
                    && execution_schedule_num_neigh_host[
                        receiver_begin+local_receiver] == tile_degree;

            for (int l=0; l<=l_max; ++l) {
                const int components = 2*l+1;
                const int group_begin = execution_group_path_offsets_host[l];
                const int num_eta =
                    execution_group_path_offsets_host[l+1]-group_begin;
                const int group_lme_begin = execution_group_lme_offsets_host[l];
                const int coupling_columns = num_eta*components*channels;
                auto state = factorized_state_view(l);
                auto output = factorized_output_view(l);
                auto projection = execution_projection(l);

                Kokkos::parallel_for(
                        "MACEKokkos::compute_factorized_forward_coupling",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                            execution_space, 0,
                            chunk_edges*coupling_columns),
                        KOKKOS_LAMBDA (const int flat) {
                            const int column = flat%coupling_columns;
                            const int local_edge = flat/coupling_columns;
                            const int edge = chunk_edge_begin+local_edge;
                            const int neighbor = neigh_indices(edge);
                            const int channel = column%channels;
                            const int eta_component = column/channels;
                            const int component = eta_component%components;
                            const int eta = eta_component/components;
                            const int lme =
                                group_lme_begin+component*num_eta+eta;
                            Precision value = 0;
                            for (int term=cg_offsets(lme);
                                 term<cg_offsets(lme+1); ++term) {
                                const int row = cg_rows(term);
                                value += cg_coefficients(term)
                                    *harmonics_values(
                                        edge*harmonics+lm1_rows(row))
                                    *neighbor_features(
                                        neighbor,lm2_rows(row),channel);
                            }
                            coupling(local_edge,column) = value;
                        });

#if (defined(KOKKOS_ENABLE_CUDA) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)) \
    || (defined(KOKKOS_ENABLE_HIP) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS))
                    if (uniform_degree && tile_degree > 0) {
                        execution_strided_batched_forward_gemm(
                            *factorized_blas_context, factorized_execution_space,
                            radial_values, coupling, aggregate,
                            receiver_count, tile_degree, embedding,
                            coupling_columns, factorized_max_coupling_columns);
                    } else
#endif
                    if (uniform_degree && tile_degree == 0) {
                        Kokkos::parallel_for(
                            "MACEKokkos::zero_factorized_forward_aggregate",
                            Kokkos::RangePolicy<
                                Kokkos::DefaultExecutionSpace>(
                                    execution_space, 0,
                                    receiver_count*embedding
                                        *coupling_columns),
                            KOKKOS_LAMBDA (const int flat) {
                                const int column = flat%coupling_columns;
                                const int q = (flat/coupling_columns)%embedding;
                                const int local_receiver =
                                    flat/(coupling_columns*embedding);
                                aggregate(local_receiver,q,column) = Precision(0);
                            });
                    } else {
                        for (int local_receiver=0;
                             local_receiver<receiver_count; ++local_receiver) {
                            const int receiver = receiver_begin+local_receiver;
                            const int local_edge_begin =
                                execution_receiver_offsets_host[receiver]
                                    -chunk_edge_begin;
                            const int local_edge_end = local_edge_begin
                                +execution_schedule_num_neigh_host[receiver];
                            const auto aggregate_receiver = Kokkos::subview(
                                aggregate, local_receiver, Kokkos::ALL,
                                Kokkos::make_pair(0, coupling_columns));
                            if (local_edge_begin == local_edge_end) {
                                Kokkos::deep_copy(
                                    execution_space, aggregate_receiver,
                                    Precision(0));
                                continue;
                            }
                            const auto radial_receiver = Kokkos::subview(
                                radial_values,
                                Kokkos::make_pair(
                                    local_edge_begin, local_edge_end),
                                Kokkos::ALL);
                            const auto coupling_receiver = Kokkos::subview(
                                coupling,
                                Kokkos::make_pair(
                                    local_edge_begin, local_edge_end),
                                Kokkos::make_pair(0, coupling_columns));
                            KokkosBlas::gemm(
                                execution_space, "T", "N", Precision(1),
                                radial_receiver,
                                coupling_receiver, Precision(0),
                                aggregate_receiver);
                        }
                    }

                if (factorized_observer_enabled)
                    capture_factorized_state_tile(
                        l, receiver_begin, receiver_count, aggregate);

                Kokkos::parallel_for(
                    "MACEKokkos::scatter_factorized_forward_state",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0,
                        receiver_count*embedding*coupling_columns),
                    KOKKOS_LAMBDA (const int flat) {
                        const int column = flat%coupling_columns;
                        const int q = (flat/coupling_columns)%embedding;
                        const int local_receiver =
                            flat/(coupling_columns*embedding);
                        const int channel = column%channels;
                        const int eta_component = column/channels;
                        const int component = eta_component%components;
                        const int eta = eta_component/components;
                        state(local_receiver*components+component,
                              (eta*embedding+q)*channels+channel) =
                            aggregate(local_receiver,q,column);
                    });

                const auto live_state = Kokkos::subview(
                    state,
                    Kokkos::make_pair(0, receiver_count*components),
                    Kokkos::ALL);
                const auto live_output = Kokkos::subview(
                    output,
                    Kokkos::make_pair(0, receiver_count*components),
                    Kokkos::ALL);
                KokkosBlas::gemm(
                    execution_space, "N", "N", Precision(1), live_state,
                    projection,
                    Precision(0), live_output);
                Kokkos::parallel_for(
                    "MACEKokkos::scatter_factorized_forward_output",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0,
                        receiver_count*components*channels),
                    KOKKOS_LAMBDA (const int flat) {
                        const int channel = flat%channels;
                        const int row = flat/channels;
                        const int component = row%components;
                        const int local_receiver = row/components;
                        output_features(
                            receiver_begin+local_receiver,
                            l*l+component,channel) = live_output(row,channel);
                    });
            }
        }
        capture_factorized_output(num_nodes);
        complete_device_stage("MACEKokkos::compute_factorized_tiled");
        return;
    }

    for (int l=0; l<=l_max; ++l) {
        const int components = 2*l+1;
        const int group_begin = execution_group_path_offsets_host[l];
        const int num_eta = execution_group_path_offsets_host[l+1]-group_begin;
        auto state = factorized_state_view(l);
        auto output = factorized_output_view(l);
        auto projection = execution_projection(l);

        for (int receiver_begin=0; receiver_begin<num_nodes;
             receiver_begin += factorized_chunk_size) {
            const int receiver_count = std::min(
                factorized_chunk_size, num_nodes-receiver_begin);
            Kokkos::deep_copy(execution_space, state, Precision(0));

            const auto scratch_bytes = admitted_team_scratch_bytes<>(
                "MACEKokkos::compute_factorized_state",
                {static_cast<std::size_t>(embedding), sizeof(Precision)});
            auto policy = Kokkos::TeamPolicy<>(
                execution_space, receiver_count*num_eta, Kokkos::AUTO, 32);
            policy.set_scratch_size(
                0, Kokkos::PerTeam(scratch_bytes));
            Kokkos::parallel_for(
                "MACEKokkos::compute_factorized_state", policy,
                KOKKOS_LAMBDA (const TeamMember& member) {
                    const int local_receiver = member.league_rank()/num_eta;
                    const int eta = member.league_rank()%num_eta;
                    const int receiver = receiver_begin+local_receiver;
                    const int path = group_paths(group_begin+eta);
                    const int type_i = type_map(node_types(receiver));
                    const int edge_begin = first_neigh(receiver);
                    ScratchView radial_values(member.team_scratch(0), embedding);

                    for (int neighbor_offset=0;
                         neighbor_offset<num_neigh(receiver); ++neighbor_offset) {
                        const int edge = edge_begin+neighbor_offset;
                        const int type_j = type_map(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        const auto point = radial.evaluation_point(r(edge));
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(member, embedding),
                            [=] (const int q) {
                                radial_values(q) = radial.evaluate_function(
                                    edge_type, point, q);
                            });
                        member.team_barrier();

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(member, components*channels),
                            [=] (const int component_channel) {
                                const int component = component_channel/channels;
                                const int channel = component_channel%channels;
                                const int lme = path_component_lme(
                                    path_component_offsets(path)+component);
                                Precision angular = 0;
                                for (int term=cg_offsets(lme);
                                     term<cg_offsets(lme+1); ++term) {
                                    const int row = cg_rows(term);
                                    angular += cg_coefficients(term)
                                        *harmonics_values(
                                            edge*harmonics+lm1_rows(row))
                                        *neighbor_features(
                                            neigh_indices(edge),lm2_rows(row),channel);
                                }
                                const int output_row =
                                    local_receiver*components+component;
                                Kokkos::parallel_for(
                                    Kokkos::ThreadVectorRange(member, embedding),
                                    [=] (const int q) {
                                        const int column =
                                            (eta*embedding+q)*channels+channel;
                                        state(output_row,column) +=
                                            radial_values(q)*angular;
                                    });
                            });
                        member.team_barrier();
                    }
                });

            if (factorized_observer_enabled)
                capture_factorized_state_tile(
                    l, receiver_begin, receiver_count, state);

            KokkosBlas::gemm(
                execution_space, "N", "N", Precision(1), state, projection,
                Precision(0), output);
            Kokkos::parallel_for(
                "MACEKokkos::scatter_factorized_output",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0,
                    receiver_count*components*channels),
                KOKKOS_LAMBDA (const int flat) {
                    const int channel = flat%channels;
                    const int row = flat/channels;
                    const int component = row%components;
                    const int local_receiver = row/components;
                    output_features(
                        receiver_begin+local_receiver,l*l+component,channel) =
                        output(row,channel);
                });
        }
    }
    capture_factorized_output(num_nodes);
    complete_device_stage("MACEKokkos::compute_factorized_stateful");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_factorized_direct(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    const auto execution_space = factorized_execution_space;
    const int num_feature_nodes = execution_schedule_num_feature_nodes;
    const bool compact_geometry = use_compact_edge_geometry();
    const bool recompute_harmonic_gradients = use_y_only_direct_harmonics();
    const auto unit_direction = execution_prepared_unit_direction;
    if (use_factorized_direct_jit_reverse()) {
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
        if (jit_device_plugin_ready()) {
            if constexpr (!device_execution_space<
                    Kokkos::DefaultExecutionSpace>) {
                throw std::runtime_error(
                    "Execution device plugins require device execution.");
            } else {
#endif
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
                static_assert(std::is_same_v<int,std::int32_t>);
                require_execution_device_packet(num_nodes >= 0, "negative node count");
                require_execution_device_packet(
                    neigh_indices.extent(0)
                        <= static_cast<std::size_t>(
                            std::numeric_limits<std::int32_t>::max()),
                    "edge count exceeds the CUDA ABI");
                const int num_edges = static_cast<int>(neigh_indices.extent(0));
                const int channels = num_channels;
                const auto radial = make_execution_device_radial_spline(radial_1);
                const int expected_edge_types =
                    num_active_types*(num_active_types+1)/2;
                const int expected_functions =
                    static_cast<int>(Phi1_l.extent(0))*channels;

                if (use_receiver_local_phi1()) {
                    require_execution_device_packet(
                        execution_prepared_num_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && streamed_first_neigh.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && A1_adj.extent(0)
                                >= static_cast<std::size_t>(num_nodes)
                            && H1.extent(0)
                                >= static_cast<std::size_t>(num_feature_nodes)
                            && H1_adj.extent(0)
                                >= static_cast<std::size_t>(num_feature_nodes),
                        "projected R1 reverse node tensor extent is too small");
                    require_execution_device_packet(
                        execution_a1_projection_weights.data() != nullptr
                            && A1_adj.span_is_contiguous()
                            && H1_adj.span_is_contiguous(),
                        "projected R1 reverse storage is invalid");
                    const ExecutionDeviceR1ProjectedReverseArgs args {
                        sizeof(ExecutionDeviceR1ProjectedReverseArgs),
                        static_cast<std::uint32_t>(num_active_types),
                        compact_geometry
                            ? static_cast<std::uint32_t>(sizeof(Precision))
                            : static_cast<std::uint32_t>(sizeof(double)),
                        compact_geometry ? 1u : 0u,
                        num_nodes,
                        num_edges,
                        reinterpret_cast<const std::int32_t*>(node_types.data()),
                        reinterpret_cast<const std::int32_t*>(
                            execution_prepared_num_neigh.data()),
                        reinterpret_cast<const std::int32_t*>(
                            streamed_first_neigh.data()),
                        reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                        reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                        reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                        compact_geometry
                            ? static_cast<const void*>(unit_direction.data())
                            : static_cast<const void*>(xyz.data()),
                        r.data(),
                        radial,
                        Y.data(),
                        Y_grad.data(),
                        H1.data(),
                        A1_adj.data(),
                        H1_adj.data(),
                        node_forces.data(),
                        execution_a1_projection_weights.data(),
                        r_cut};
                    const int persistent_blocks =
                        resolve_jit_device_plugin_persistent_blocks(
                            *jit_device_plugin,
                            execution_space,
                            static_cast<std::size_t>(num_nodes),
                            "r1_projected_reverse");
                    ExecutionDeviceBackend::DeviceGuard device_guard(
                        ExecutionDeviceBackend::device_ordinal(execution_space));
                    ExecutionDeviceBackend::check_status(
                        jit_device_plugin->launch_r1_projected_reverse(
                            &args,
                            reinterpret_cast<void*>(
                                ExecutionDeviceBackend::native_stream(execution_space)),
                            persistent_blocks),
                        "Launching the projected R1 reverse module");
                    if (num_edges > 0) {
                        factorized_jit_launch_count += 1;
                        factorized_jit_reverse_launch_count += 1;
                    }
                    return;
                }

                if (use_channel_tiled_phi1()) {
                    const bool receiver_tiled = dual_layer_tiled_plan_active;
                    Kokkos::View<int*> source_offsets;
                    Kokkos::View<int*> source_edges;
                    Kokkos::View<int*> edge_receivers;
                    Kokkos::View<int*> source_ids;
                    if (receiver_tiled) {
                        const std::size_t segment_begin =
                            static_cast<std::size_t>(
                                dual_layer_active_segment_begin);
                        const std::size_t segment_count =
                            static_cast<std::size_t>(
                                dual_layer_active_segment_count);
                        const std::size_t segment_end =
                            segment_begin+segment_count;
                        source_offsets = Kokkos::subview(
                            dual_layer_segment_edge_offsets,
                            Kokkos::make_pair(
                                segment_begin, segment_end+std::size_t(1)));
                        source_edges = dual_layer_source_edges;
                        edge_receivers = Kokkos::subview(
                            dual_layer_edge_local_receivers,
                            Kokkos::make_pair(
                                std::size_t(0),
                                static_cast<std::size_t>(num_edges)));
                        source_ids = Kokkos::subview(
                            dual_layer_segment_source_ids,
                            Kokkos::make_pair(segment_begin, segment_end));
                    } else {
                        source_offsets = execution_direct_source_offsets;
                        source_edges = execution_direct_source_edges;
                        edge_receivers = execution_edge_receivers;
                    }
                    const std::size_t required_source_offsets = receiver_tiled
                        ? static_cast<std::size_t>(
                            dual_layer_active_segment_count)+std::size_t(1)
                        : static_cast<std::size_t>(num_feature_nodes)
                            +std::size_t(1);
                    require_execution_device_packet(
                        radial.edge_types == static_cast<std::uint32_t>(
                            expected_edge_types)
                            && radial.functions == static_cast<std::uint32_t>(
                                expected_functions)
                            && radial.intervals > 0 && radial.h > 0.0
                            && std::isfinite(r_cut) && r_cut > 0.0
                            && radial.coefficients != nullptr,
                        "tiled R1 reverse radial spline storage is invalid");
                    require_execution_device_packet(
                        Phi1.extent(0) >= static_cast<std::size_t>(num_nodes)
                            && Phi1.extent_int(1) == num_lme
                            && Phi1.extent_int(2) == phi1_channel_tile_size
                            && H1.extent(0)
                                >= static_cast<std::size_t>(num_feature_nodes)
                            && H1_adj.extent(0)
                                >= static_cast<std::size_t>(num_feature_nodes),
                        "tiled R1 reverse node tensor shape is inconsistent");
                    require_execution_device_packet(
                        source_offsets.extent(0)
                                >= required_source_offsets
                            && source_edges.extent(0)
                                >= static_cast<std::size_t>(num_edges)
                            && edge_receivers.extent(0)
                                >= static_cast<std::size_t>(num_edges),
                        "tiled R1 reverse source schedule is incomplete");
                    require_execution_device_packet(
                        Phi1.span_is_contiguous()
                            && H1.span_is_contiguous()
                            && H1_adj.span_is_contiguous()
                            && node_forces.span_is_contiguous(),
                        "tiled R1 reverse storage is not contiguous");
                    const int persistent_blocks =
                        resolve_jit_device_plugin_persistent_blocks(
                            *jit_device_plugin,
                            execution_space,
                            static_cast<std::size_t>(num_feature_nodes),
                            "r1_tiled_reverse");
                    ExecutionDeviceBackend::DeviceGuard device_guard(
                        ExecutionDeviceBackend::device_ordinal(execution_space));
                    for (int channel_begin=0; channel_begin<channels;
                         channel_begin += phi1_channel_tile_size) {
                        const int channel_count = std::min(
                            phi1_channel_tile_size, channels-channel_begin);
                        reverse_A1_channel_tile(num_nodes, channel_begin);
                        const ExecutionDeviceR1TiledSourceArgs source_args {
                            sizeof(ExecutionDeviceR1TiledSourceArgs),
                            static_cast<std::uint32_t>(num_active_types),
                            static_cast<std::uint32_t>(channel_begin),
                            static_cast<std::uint32_t>(channel_count),
                            num_feature_nodes,
                            num_edges,
                            reinterpret_cast<const std::int32_t*>(node_types.data()),
                            reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                            reinterpret_cast<const std::int32_t*>(
                                source_offsets.data()),
                            reinterpret_cast<const std::int32_t*>(
                                source_edges.data()),
                            reinterpret_cast<const std::int32_t*>(
                                edge_receivers.data()),
                            reinterpret_cast<const std::int32_t*>(
                                type_to_active.data()),
                            r.data(),
                            radial,
                            Y.data(),
                            Phi1.data(),
                            H1_adj.data(),
                            r_cut,
                            receiver_tiled
                                ? dual_layer_active_segment_count : 0,
                            receiver_tiled
                                ? reinterpret_cast<const std::int32_t*>(
                                    source_ids.data())
                                : nullptr};
                        const ExecutionDeviceR1TiledEdgeArgs edge_args {
                            sizeof(ExecutionDeviceR1TiledEdgeArgs),
                            static_cast<std::uint32_t>(num_active_types),
                            compact_geometry
                                ? static_cast<std::uint32_t>(sizeof(Precision))
                                : static_cast<std::uint32_t>(sizeof(double)),
                            compact_geometry ? 1u : 0u,
                            num_nodes,
                            num_edges,
                            reinterpret_cast<const std::int32_t*>(node_types.data()),
                            reinterpret_cast<const std::int32_t*>(
                                neigh_indices.data()),
                            reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                            reinterpret_cast<const std::int32_t*>(
                                edge_receivers.data()),
                            reinterpret_cast<const std::int32_t*>(
                                type_to_active.data()),
                            compact_geometry
                                ? static_cast<const void*>(unit_direction.data())
                                : static_cast<const void*>(xyz.data()),
                            r.data(),
                            radial,
                            Y.data(),
                            Y_grad.data(),
                            H1.data(),
                            Phi1.data(),
                            node_forces.data(),
                            r_cut};
                        ExecutionDeviceBackend::check_status(
                            jit_device_plugin->launch_r1_tiled_reverse(
                                &source_args,
                                &edge_args,
                                reinterpret_cast<void*>(
                                    ExecutionDeviceBackend::native_stream(
                                        execution_space)),
                                persistent_blocks),
                            "Launching the tiled R1 reverse module");
                        if (num_edges > 0) {
                            factorized_jit_launch_count += 1;
                            factorized_jit_reverse_launch_count += 1;
                        }
                    }
                    return;
                }

                require_execution_device_packet(
                    radial.edge_types == static_cast<std::uint32_t>(
                        expected_edge_types),
                    "radial edge-type extent does not match active types");
                require_execution_device_packet(
                    radial.functions == static_cast<std::uint32_t>(
                        expected_functions),
                    "radial function extent does not match R1 paths");
                require_execution_device_packet(
                    radial.intervals > 0 && radial.h > 0.0
                        && std::isfinite(r_cut) && r_cut > 0.0
                        && radial.coefficients != nullptr,
                    "radial spline storage is invalid");
                require_execution_device_packet(
                    node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && H1.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)
                        && H1_adj.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)
                        && dPhi1.extent(0)
                            >= static_cast<std::size_t>(num_nodes),
                    "node tensor extent is too small");
                require_execution_device_packet(
                    neigh_types.extent(0) >= static_cast<std::size_t>(num_edges)
                        && execution_edge_receivers.extent(0)
                            >= static_cast<std::size_t>(num_edges)
                        && (compact_geometry
                            ? unit_direction.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges
                            : xyz.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges)
                        && r.extent(0) >= static_cast<std::size_t>(num_edges)
                        && node_forces.extent(0)
                            >= static_cast<std::size_t>(3)*num_edges,
                    "edge tensor extent is too small");
                require_execution_device_packet(
                    execution_direct_source_offsets.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)+1
                        && execution_direct_source_edges.extent(0)
                            >= static_cast<std::size_t>(num_edges),
                    "source schedule extent is too small");
                require_execution_device_packet(
                    type_to_active.extent(0)
                        >= static_cast<std::size_t>(num_elements),
                    "type map extent is too small");
                require_execution_device_packet(
                    Y.extent(0) >= static_cast<std::size_t>(num_edges)*num_lm
                        && (recompute_harmonic_gradients
                            || Y_grad.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges*num_lm),
                    "harmonic tensor extent is too small");
                require_execution_device_packet(
                    H1.extent(1) >= static_cast<std::size_t>(num_LM)
                        && H1.extent(2) >= static_cast<std::size_t>(channels)
                        && H1_adj.extent(1)
                            >= static_cast<std::size_t>(num_LM)
                        && H1_adj.extent(2)
                            >= static_cast<std::size_t>(channels)
                        && dPhi1.extent(1)
                            >= static_cast<std::size_t>(num_lme)
                        && dPhi1.extent(2)
                            >= static_cast<std::size_t>(channels),
                    "R1 tensor shape is inconsistent with the plugin contract");
                require_execution_device_packet(
                    num_nodes == 0
                        || (node_types.data() != nullptr
                            && execution_direct_source_offsets.data() != nullptr
                            && type_to_active.data() != nullptr
                            && H1.data() != nullptr
                            && dPhi1.data() != nullptr
                            && H1_adj.data() != nullptr),
                    "required node tensor pointer is null");
                require_execution_device_packet(
                    num_edges == 0
                        || (neigh_indices.data() != nullptr
                            && neigh_types.data() != nullptr
                            && execution_direct_source_edges.data() != nullptr
                            && execution_edge_receivers.data() != nullptr
                            && (compact_geometry
                                ? unit_direction.data() != nullptr
                                : xyz.data() != nullptr)
                            && r.data() != nullptr
                            && Y.data() != nullptr
                            && (recompute_harmonic_gradients
                                || Y_grad.data() != nullptr)
                            && node_forces.data() != nullptr),
                    "required edge tensor pointer is null");
                require_execution_device_packet(
                    node_types.span_is_contiguous()
                        && neigh_indices.span_is_contiguous()
                        && neigh_types.span_is_contiguous()
                        && execution_direct_source_offsets.span_is_contiguous()
                        && execution_direct_source_edges.span_is_contiguous()
                        && execution_edge_receivers.span_is_contiguous()
                        && type_to_active.span_is_contiguous()
                        && (compact_geometry
                            ? unit_direction.span_is_contiguous()
                            : xyz.span_is_contiguous())
                        && r.span_is_contiguous()
                        && Y.span_is_contiguous()
                        && (recompute_harmonic_gradients
                            || Y_grad.span_is_contiguous())
                        && H1.span_is_contiguous()
                        && dPhi1.span_is_contiguous()
                        && H1_adj.span_is_contiguous()
                        && node_forces.span_is_contiguous(),
                    "tensor storage is not contiguous");

                const ExecutionDeviceR1SourceArgs source_args {
                    sizeof(ExecutionDeviceR1SourceArgs),
                    static_cast<std::uint32_t>(num_active_types),
                    num_feature_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_direct_source_offsets.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_direct_source_edges.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_edge_receivers.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    dPhi1.data(),
                    H1_adj.data(),
                    r_cut};
                const ExecutionDeviceR1EdgeArgs edge_args {
                    sizeof(ExecutionDeviceR1EdgeArgs),
                    static_cast<std::uint32_t>(num_active_types),
                    compact_geometry
                        ? static_cast<std::uint32_t>(sizeof(Precision))
                        : static_cast<std::uint32_t>(sizeof(double)),
                    compact_geometry ? 1u : 0u,
                    num_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_edge_receivers.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    compact_geometry
                        ? static_cast<const void*>(unit_direction.data())
                        : static_cast<const void*>(xyz.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    Y_grad.data(),
                    H1.data(),
                    dPhi1.data(),
                    node_forces.data(),
                    r_cut};
                const int persistent_blocks =
                    resolve_jit_device_plugin_persistent_blocks(
                        *jit_device_plugin,
                        execution_space,
                        std::max(
                            static_cast<std::size_t>(num_feature_nodes)*num_channels,
                            neigh_indices.extent(0)),
                        "r1_reverse");
                ExecutionDeviceBackend::DeviceGuard device_guard(
                    ExecutionDeviceBackend::device_ordinal(execution_space));
                ExecutionDeviceBackend::check_status(
                    jit_device_plugin->launch_r1_coordinate_reverse(
                        &source_args,
                        &edge_args,
                        reinterpret_cast<void*>(
                            ExecutionDeviceBackend::native_stream(execution_space)),
                        persistent_blocks),
                    "Launching the Execution device R1 coordinate-reverse plugin");
                if (num_edges > 0) {
                    const auto launch_count =
                        jit_device_plugin->r1_reverse_physical_launch_count();
                    factorized_jit_launch_count += launch_count;
                    factorized_jit_reverse_launch_count += launch_count;
                }
                return;
            }
        }
#endif
        if (jit_host_plugin_ready()) {
            if constexpr (!std::is_same_v<
                    typename Kokkos::DefaultExecutionSpace::memory_space,
                    Kokkos::HostSpace>) {
                throw std::runtime_error(
                    "Execution host plugins require host execution.");
            } else {
                static_assert(std::is_same_v<int,std::int32_t>);
                const int num_edges = static_cast<int>(neigh_indices.extent(0));
                const int channels = num_channels;
                const auto radial = make_execution_host_radial_spline(radial_1);
                const int expected_edge_types =
                    num_active_types*(num_active_types+1)/2;
                const int expected_functions =
                    static_cast<int>(Phi1_l.extent(0))*channels;

                require_execution_host_packet(num_nodes >= 0, "negative node count");
                require_execution_host_packet(
                    num_edges >= 0, "edge count exceeds the host ABI");
                require_execution_host_packet(
                    radial.edge_types == static_cast<std::uint32_t>(
                        expected_edge_types),
                    "radial edge-type extent does not match active types");
                require_execution_host_packet(
                    radial.functions == static_cast<std::uint32_t>(
                        expected_functions),
                    "radial function extent does not match R1 paths");
                require_execution_host_packet(
                    radial.intervals > 0 && radial.h > 0.0
                        && std::isfinite(r_cut) && r_cut > 0.0,
                    "radial spline has no positive interval grid");
                require_execution_host_packet(
                    radial.coefficients != nullptr,
                    "radial coefficient storage is null");
                require_execution_host_packet(
                    node_types.extent(0) >= static_cast<std::size_t>(num_nodes)
                        && H1.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)
                        && H1_adj.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)
                        && dPhi1.extent(0)
                            >= static_cast<std::size_t>(num_nodes),
                    "node tensor extent is too small");
                require_execution_host_packet(
                    neigh_types.extent(0) >= static_cast<std::size_t>(num_edges)
                        && execution_edge_receivers.extent(0)
                            >= static_cast<std::size_t>(num_edges)
                        && (compact_geometry
                            ? unit_direction.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges
                            : xyz.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges)
                        && r.extent(0) >= static_cast<std::size_t>(num_edges)
                        && node_forces.extent(0)
                            >= static_cast<std::size_t>(3)*num_edges,
                    "edge tensor extent is too small");
                require_execution_host_packet(
                    execution_direct_source_offsets.extent(0)
                            >= static_cast<std::size_t>(num_feature_nodes)+1
                        && execution_direct_source_edges.extent(0)
                            >= static_cast<std::size_t>(num_edges),
                    "source schedule extent is too small");
                require_execution_host_packet(
                    type_to_active.extent(0)
                        >= static_cast<std::size_t>(num_elements),
                    "type map extent is too small");
                require_execution_host_packet(
                    Y.extent(0) >= static_cast<std::size_t>(num_edges)*num_lm
                        && (recompute_harmonic_gradients
                            || Y_grad.extent(0)
                                >= static_cast<std::size_t>(3)*num_edges*num_lm),
                    "harmonic tensor extent is too small");
                require_execution_host_packet(
                    H1.extent(1) >= static_cast<std::size_t>(num_LM)
                        && H1.extent(2) >= static_cast<std::size_t>(channels)
                        && H1_adj.extent(1)
                            >= static_cast<std::size_t>(num_LM)
                        && H1_adj.extent(2)
                            >= static_cast<std::size_t>(channels)
                        && dPhi1.extent(1)
                            >= static_cast<std::size_t>(num_lme)
                        && dPhi1.extent(2)
                            >= static_cast<std::size_t>(channels),
                    "R1 tensor shape is inconsistent with the plugin contract");
                require_execution_host_packet(
                    num_nodes == 0
                        || (node_types.data() != nullptr
                            && execution_direct_source_offsets.data() != nullptr
                            && type_to_active.data() != nullptr
                            && H1.data() != nullptr
                            && dPhi1.data() != nullptr
                            && H1_adj.data() != nullptr),
                    "required node tensor pointer is null");
                require_execution_host_packet(
                    num_edges == 0
                        || (neigh_indices.data() != nullptr
                            && neigh_types.data() != nullptr
                            && execution_direct_source_edges.data() != nullptr
                            && execution_edge_receivers.data() != nullptr
                            && (compact_geometry
                                ? unit_direction.data() != nullptr
                                : xyz.data() != nullptr)
                            && r.data() != nullptr
                            && Y.data() != nullptr
                            && (recompute_harmonic_gradients
                                || Y_grad.data() != nullptr)
                            && node_forces.data() != nullptr),
                    "required edge tensor pointer is null");
                require_execution_host_packet(
                    node_types.span_is_contiguous()
                        && neigh_indices.span_is_contiguous()
                        && neigh_types.span_is_contiguous()
                        && execution_direct_source_offsets.span_is_contiguous()
                        && execution_direct_source_edges.span_is_contiguous()
                        && execution_edge_receivers.span_is_contiguous()
                        && type_to_active.span_is_contiguous()
                        && (compact_geometry
                            ? unit_direction.span_is_contiguous()
                            : xyz.span_is_contiguous())
                        && r.span_is_contiguous()
                        && Y.span_is_contiguous()
                        && (recompute_harmonic_gradients
                            || Y_grad.span_is_contiguous())
                        && H1.span_is_contiguous()
                        && dPhi1.span_is_contiguous()
                        && H1_adj.span_is_contiguous()
                        && node_forces.span_is_contiguous(),
                    "tensor storage is not contiguous");

                const SymmetrixJitHostR1SourceArgsV2 source_args {
                    sizeof(SymmetrixJitHostR1SourceArgsV2),
                    static_cast<std::uint32_t>(num_active_types),
                    num_feature_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_direct_source_offsets.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_direct_source_edges.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_edge_receivers.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    dPhi1.data(),
                    H1_adj.data(),
                    r_cut};
                const SymmetrixJitHostR1EdgeArgsV2 edge_args {
                    sizeof(SymmetrixJitHostR1EdgeArgsV2),
                    static_cast<std::uint32_t>(num_active_types),
                    compact_geometry
                        ? static_cast<std::uint32_t>(sizeof(Precision))
                        : static_cast<std::uint32_t>(sizeof(double)),
                    compact_geometry ? 1u : 0u,
                    num_nodes,
                    num_edges,
                    reinterpret_cast<const std::int32_t*>(node_types.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_indices.data()),
                    reinterpret_cast<const std::int32_t*>(neigh_types.data()),
                    reinterpret_cast<const std::int32_t*>(
                        execution_edge_receivers.data()),
                    reinterpret_cast<const std::int32_t*>(type_to_active.data()),
                    compact_geometry
                        ? static_cast<const void*>(unit_direction.data())
                        : static_cast<const void*>(xyz.data()),
                    r.data(),
                    radial,
                    Y.data(),
                    Y_grad.data(),
                    H1.data(),
                    dPhi1.data(),
                    node_forces.data(),
                    r_cut};
                const auto& descriptor = jit_host_plugin->descriptor_v2();
                const auto source_owner =
                    descriptor.r1_compensated_source_owner;
                const auto edge_owner = descriptor.r1_edge_owner;
                constexpr int source_channel_tile = 32;
                const int source_channel_tiles =
                    (channels+source_channel_tile-1)/source_channel_tile;
                Kokkos::parallel_for(
                    "ExecutionHostPlugin::r1_source",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0,
                        num_feature_nodes*source_channel_tiles),
                    [=] (const int flat) {
                        source_owner(
                            &source_args,
                            flat/source_channel_tiles,
                            (flat%source_channel_tiles)*source_channel_tile);
                    });
                Kokkos::parallel_for(
                    "ExecutionHostPlugin::r1_edge",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0, num_edges),
                    [=] (const int edge) { edge_owner(&edge_args, edge); });
                if (num_edges > 0) {
                    factorized_jit_launch_count += 2;
                    factorized_jit_reverse_launch_count += 2;
                }
                return;
            }
        }
        throw std::runtime_error(
            "Execution R1 JIT reverse selected without a loaded plugin.");
    }

    if (dPhi1r.extent(0) < static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(dPhi1r, num_nodes, num_lelm1lm2, num_channels);
    Kokkos::deep_copy(execution_space, dPhi1r, Precision(0));

    const int channels = num_channels;
    const int harmonics = num_lm;
    const int paths = static_cast<int>(Phi1_l.extent(0));
    const int source_harmonics = static_cast<int>(H1.extent(1));
    if (source_harmonics > streamed_fused_max_num_LM)
        throw std::runtime_error(
            "Execution direct source harmonics exceed the fused kernel limit.");
    const int active_type_count = num_active_types;
    const auto lme = Phi1_lme;
    const auto row = Phi1_lelm1lm2;
    const auto coefficients = Phi1_clebsch_gordan;
    const auto output_adjoint = dPhi1;
    auto row_adjoint = dPhi1r;

    const auto artifact_environment =
        kernel_launch_environment(execution_space);
    if (artifact_environment.backend == "host") {
        Kokkos::parallel_for(
            "ExecutionDirect::cg_transpose_host",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                execution_space, 0, num_nodes*channels),
            KOKKOS_LAMBDA (const int flat) {
                const int receiver = flat/channels;
                const int channel = flat%channels;
                for (int term=0; term<coefficients.extent_int(0); ++term)
                    row_adjoint(receiver,row(term),channel) += coefficients(term)
                        *output_adjoint(receiver,lme(term),channel);
            });
    } else {
        Kokkos::parallel_for(
            "ExecutionDirect::cg_transpose",
            Kokkos::TeamPolicy<>(
                execution_space, num_nodes, Kokkos::AUTO, 32),
            KOKKOS_LAMBDA (Kokkos::TeamPolicy<>::member_type member) {
                const int receiver = member.league_rank();
                for (int term=0; term<coefficients.extent_int(0); ++term) {
                    const Precision coefficient = coefficients(term);
                    Kokkos::parallel_for(
                        Kokkos::TeamVectorRange(member, channels),
                        [=] (const int channel) {
                            row_adjoint(receiver,row(term),channel) += coefficient
                                *output_adjoint(receiver,lme(term),channel);
                        });
                }
            });
    }

    const auto source_offsets = execution_direct_source_offsets;
    const auto source_edges = execution_direct_source_edges;
    const auto edge_receivers = execution_edge_receivers;
    const auto type_map = type_to_active;
    const auto path_row_offsets = Phi1_path_row_offsets;
    const auto lm1_rows = Phi1_lm1;
    const auto lm2_rows = Phi1_lm2;
    const auto radial = radial_1;
    const auto harmonics_values = Y;
    auto source_adjoint = H1_adj;

    Kokkos::parallel_for(
        "ExecutionDirect::source_transpose",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            execution_space, 0, num_feature_nodes*channels),
        KOKKOS_LAMBDA (const int flat) {
            const int channel = flat%channels;
            const int source = flat/channels;
            Precision values[streamed_fused_max_num_LM] = {};
            Precision compensations[streamed_fused_max_num_LM] = {};
            for (int scheduled=source_offsets(source);
                 scheduled<source_offsets(source+1); ++scheduled) {
                const int edge = source_edges(scheduled);
                const int receiver = edge_receivers(edge);
                const int receiver_type = type_map(node_types(receiver));
                const int source_type = type_map(neigh_types(edge));
                const int edge_type = receiver_type <= source_type
                    ? receiver_type*(2*active_type_count-receiver_type-1)/2
                        +source_type
                    : source_type*(2*active_type_count-source_type-1)/2
                        +receiver_type;
                const auto point = radial.evaluation_point(r(edge));
                for (int path=0; path<paths; ++path) {
                    const Precision radial_value = radial.evaluate_function(
                        edge_type, point, path*channels+channel);
                    for (int sparse_row=path_row_offsets(path);
                         sparse_row<path_row_offsets(path+1); ++sparse_row) {
                        const int source_lm = lm2_rows(sparse_row);
                        const Precision contribution = radial_value
                            *harmonics_values(
                                edge*harmonics+lm1_rows(sparse_row))
                            *row_adjoint(receiver,sparse_row,channel);
                        const Precision corrected = contribution
                            -compensations[source_lm];
                        const Precision updated = values[source_lm]+corrected;
                        compensations[source_lm] =
                            (updated-values[source_lm])-corrected;
                        values[source_lm] = updated;
                    }
                }
            }
            for (int source_lm=0; source_lm<source_harmonics; ++source_lm)
                source_adjoint(source,source_lm,channel) += values[source_lm];
        });

    const auto harmonics_gradients = Y_grad;
    const auto neighbor_features = H1;
    auto forces = node_forces;
    const int num_edges = static_cast<int>(neigh_indices.extent(0));
    if (num_edges > 0) {
        if (artifact_environment.backend == "host") {
            Kokkos::parallel_for(
                "ExecutionDirect::edge_force_host",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    execution_space, 0, num_edges),
                KOKKOS_LAMBDA (const int edge) {
                    const int receiver = edge_receivers(edge);
                    const int source = neigh_indices(edge);
                    const int receiver_type = type_map(node_types(receiver));
                    const int source_type = type_map(neigh_types(edge));
                    const int edge_type = receiver_type <= source_type
                        ? receiver_type
                            *(2*active_type_count-receiver_type-1)/2
                            +source_type
                        : source_type
                            *(2*active_type_count-source_type-1)/2
                            +receiver_type;
                    const auto point = radial.evaluation_point(r(edge));
                    const Precision x_over_r = compact_geometry
                        ? unit_direction(3*edge)
                        : static_cast<Precision>(xyz(3*edge)/r(edge));
                    const Precision y_over_r = compact_geometry
                        ? unit_direction(3*edge+1)
                        : static_cast<Precision>(xyz(3*edge+1)/r(edge));
                    const Precision z_over_r = compact_geometry
                        ? unit_direction(3*edge+2)
                        : static_cast<Precision>(xyz(3*edge+2)/r(edge));
                    Precision force_x = 0;
                    Precision force_y = 0;
                    Precision force_z = 0;
                    for (int channel=0; channel<channels; ++channel)
                        for (int path=0; path<paths; ++path) {
                            Precision radial_value;
                            Precision radial_derivative;
                            radial.evaluate_function(
                                edge_type, point, path*channels+channel,
                                radial_value, radial_derivative);
                            for (int sparse_row=path_row_offsets(path);
                                 sparse_row<path_row_offsets(path+1);
                                 ++sparse_row) {
                                const int lm1 = lm1_rows(sparse_row);
                                const int lm2 = lm2_rows(sparse_row);
                                const Precision adjoint = row_adjoint(
                                    receiver,sparse_row,channel);
                                const Precision neighbor =
                                    neighbor_features(source,lm2,channel);
                                const Precision radial_force =
                                    radial_derivative*neighbor*adjoint;
                                const Precision angular_force =
                                    radial_value*neighbor*adjoint;
                                force_x += radial_force*x_over_r
                                    *harmonics_values(edge*harmonics+lm1)
                                    +angular_force*harmonics_gradients(
                                        3*edge*harmonics+lm1);
                                force_y += radial_force*y_over_r
                                    *harmonics_values(edge*harmonics+lm1)
                                    +angular_force*harmonics_gradients(
                                        (3*edge+1)*harmonics+lm1);
                                force_z += radial_force*z_over_r
                                    *harmonics_values(edge*harmonics+lm1)
                                    +angular_force*harmonics_gradients(
                                        (3*edge+2)*harmonics+lm1);
                            }
                        }
                    forces(3*edge) -= force_x;
                    forces(3*edge+1) -= force_y;
                    forces(3*edge+2) -= force_z;
                });
        } else {
            constexpr int edges_per_team = 8;
            using TeamMember = Kokkos::TeamPolicy<>::member_type;
            Kokkos::parallel_for(
                "ExecutionDirect::edge_force",
                Kokkos::TeamPolicy<>(
                    execution_space,
                    (num_edges+edges_per_team-1)/edges_per_team,
                    edges_per_team, 32),
                KOKKOS_LAMBDA (TeamMember member) {
                const int edge_begin = member.league_rank()*edges_per_team;
                const int edge_count = Kokkos::min(
                    edges_per_team, num_edges-edge_begin);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(member, edge_count),
                    [=] (const int edge_offset) {
                        const int edge = edge_begin+edge_offset;
                        const int receiver = edge_receivers(edge);
                        const int source = neigh_indices(edge);
                        const int receiver_type = type_map(node_types(receiver));
                        const int source_type = type_map(neigh_types(edge));
                        const int edge_type = receiver_type <= source_type
                            ? receiver_type
                                *(2*active_type_count-receiver_type-1)/2
                                +source_type
                            : source_type
                                *(2*active_type_count-source_type-1)/2
                                +receiver_type;
                        const auto point = radial.evaluation_point(r(edge));
                        const Precision x_over_r = compact_geometry
                            ? unit_direction(3*edge)
                            : static_cast<Precision>(xyz(3*edge)/r(edge));
                        const Precision y_over_r = compact_geometry
                            ? unit_direction(3*edge+1)
                            : static_cast<Precision>(xyz(3*edge+1)/r(edge));
                        const Precision z_over_r = compact_geometry
                            ? unit_direction(3*edge+2)
                            : static_cast<Precision>(xyz(3*edge+2)/r(edge));
                        Precision force_x, force_y, force_z;
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(member, channels),
                            [=] (const int channel,
                                 Precision& force_x,
                                 Precision& force_y,
                                 Precision& force_z) {
                                for (int path=0; path<paths; ++path) {
                                    Precision radial_value;
                                    Precision radial_derivative;
                                    radial.evaluate_function(
                                        edge_type, point,
                                        path*channels+channel,
                                        radial_value, radial_derivative);
                                    for (int sparse_row=path_row_offsets(path);
                                         sparse_row<path_row_offsets(path+1);
                                         ++sparse_row) {
                                        const int lm1 = lm1_rows(sparse_row);
                                        const int lm2 = lm2_rows(sparse_row);
                                        const Precision adjoint = row_adjoint(
                                            receiver,sparse_row,channel);
                                        const Precision neighbor =
                                            neighbor_features(source,lm2,channel);
                                        const Precision radial_force =
                                            radial_derivative*neighbor*adjoint;
                                        const Precision angular_force =
                                            radial_value*neighbor*adjoint;
                                        force_x += radial_force*x_over_r
                                            *harmonics_values(
                                                edge*harmonics+lm1)
                                            +angular_force
                                                *harmonics_gradients(
                                                    3*edge*harmonics+lm1);
                                        force_y += radial_force*y_over_r
                                            *harmonics_values(
                                                edge*harmonics+lm1)
                                            +angular_force
                                                *harmonics_gradients(
                                                    (3*edge+1)*harmonics+lm1);
                                        force_z += radial_force*z_over_r
                                            *harmonics_values(
                                                edge*harmonics+lm1)
                                            +angular_force
                                                *harmonics_gradients(
                                                    (3*edge+2)*harmonics+lm1);
                                    }
                                }
                            }, force_x, force_y, force_z);
                        Kokkos::single(Kokkos::PerThread(member), [=] {
                            forces(3*edge) -= force_x;
                            forces(3*edge+1) -= force_y;
                            forces(3*edge+2) -= force_z;
                        });
                    });
                });
        }
    }
    complete_device_stage("MACEKokkos::reverse_factorized_direct");
}

template <typename Precision>
void MACEKokkos<Precision>::reverse_factorized(
    int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r)
{
    if (!factorized_ready)
        throw std::runtime_error("Execution R1 model cache has not been prepared.");
    const auto execution_space = factorized_execution_space;
    const int num_feature_nodes = execution_schedule_num_feature_nodes;

    using TeamMember = Kokkos::TeamPolicy<>::member_type;
    using ScratchView = Kokkos::View<
        Precision*, typename TeamMember::scratch_memory_space,
        Kokkos::MemoryUnmanaged>;
    const int channels = num_channels;
    const int harmonics = num_lm;
    const int embedding = factorized_embedding_width;
    const int active_type_count = num_active_types;
    const auto first_neigh = streamed_first_neigh;
    const auto type_map = type_to_active;
    const auto radial = execution_radial_1;
    const auto group_paths = execution_group_paths;
    const auto path_component_offsets = execution_path_component_offsets;
    const auto path_component_lme = execution_path_component_lme;
    const auto cg_offsets = execution_cg_offsets;
    const auto cg_rows = execution_cg_rows;
    const auto cg_coefficients = execution_cg_coefficients;
    const auto source_lm_offsets = execution_source_lm_offsets;
    const auto source_eta_components = execution_source_eta_components;
    const auto source_lm1 = execution_source_lm1;
    const auto source_coefficients = execution_source_coefficients;
    const auto path_row_offsets = Phi1_path_row_offsets;
    const auto row_cg_offsets = execution_row_cg_offsets;
    const auto row_cg_lme = execution_row_cg_lme;
    const auto row_cg_coefficients = execution_row_cg_coefficients;
    const auto lme_component = execution_lme_component;
    const auto lm1_rows = Phi1_lm1;
    const auto lm2_rows = Phi1_lm2;
    const auto harmonics_values = Y;
    const auto harmonics_gradients = Y_grad;
    const auto neighbor_features = H1;
    const auto output_adjoint = A1_adj;
    const bool compact_geometry = use_compact_edge_geometry();
    const auto unit_direction = execution_prepared_unit_direction;
    auto forces = node_forces;
    const auto source_chunk_offsets = execution_source_chunk_offsets;
    const auto source_edges = execution_source_edges;
    const auto edge_receivers = execution_edge_receivers;
    auto radial_values = execution_radial_values;
    auto radial_derivatives = execution_radial_derivatives;
    auto packed_state_adjoint = factorized_packed_state_view();
    auto coupling_adjoint = execution_coupling_adjoint;
    auto neighbor_adjoint = H1_adj;
    auto source_compensation = factorized_source_compensation_view();
    const int neighbor_harmonics = num_LM;
    auto compact_message = factorized_compact_message_view();
    const bool plugin_tiled_fallback =
        factorized_source_strategy == FactorizedSourceStrategy::jit_plugin
        && factorized_tiled_ready;
    const bool tiled_execution =
        factorized_source_strategy == FactorizedSourceStrategy::tiled_coupling
        || plugin_tiled_fallback;

    if (execution_schedule_num_nodes != num_nodes
        || static_cast<int>(source_edges.extent(0))
            != static_cast<int>(neigh_indices.extent(0)))
        throw std::runtime_error("Execution R1 source schedule has not been prepared.");
    begin_factorized_reverse_observation(
        num_nodes, static_cast<int>(neigh_indices.extent(0)));
    Kokkos::deep_copy(execution_space, source_compensation, Precision(0));

    const int num_chunks =
        (num_nodes+factorized_chunk_size-1)/factorized_chunk_size;
    const int num_groups = l_max+1;
    const int reverse_iterations = num_chunks*num_groups;
    for (int iteration=0; iteration<reverse_iterations; ++iteration) {
        // The admitted path keeps one radial cache live across every irrep in a
        // receiver chunk. Legacy paths retain their original irrep-outer order.
        const int l = tiled_execution
            ? iteration%num_groups : iteration/num_chunks;
        const int chunk = tiled_execution
            ? iteration/num_groups : iteration%num_chunks;
        const int receiver_begin = chunk*factorized_chunk_size;
        const int components = 2*l+1;
        const int group_begin = execution_group_path_offsets_host[l];
        const int num_eta = execution_group_path_offsets_host[l+1]-group_begin;
        const int group_lme_begin = execution_group_lme_offsets_host[l];
        auto state_adjoint = factorized_state_view(l);
        auto packed_output_adjoint = factorized_output_view(l);
        auto projection_trans = execution_projection_trans(l);

            const int receiver_count = std::min(
                factorized_chunk_size, num_nodes-receiver_begin);
            const int chunk_edge_begin = execution_chunk_edge_offsets_host[chunk];
            const int chunk_edge_end = execution_chunk_edge_offsets_host[chunk+1];
            const int chunk_edges = chunk_edge_end-chunk_edge_begin;
        const int coupling_columns = num_eta*components*channels;
            Kokkos::deep_copy(
                    execution_space, packed_output_adjoint, Precision(0));
                Kokkos::parallel_for(
                    "MACEKokkos::pack_factorized_output_adjoint",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0,
                        receiver_count*components*channels),
                    KOKKOS_LAMBDA (const int flat) {
                        const int channel = flat%channels;
                        const int row = flat/channels;
                        const int component = row%components;
                        const int local_receiver = row/components;
                        packed_output_adjoint(row,channel) = output_adjoint(
                            receiver_begin+local_receiver,
                            l*l+component,channel);
                    });
                KokkosBlas::gemm(
                    execution_space, "N", "N", Precision(1),
                    packed_output_adjoint,
                    projection_trans, Precision(0), state_adjoint);

            if (factorized_observer_enabled && !tiled_execution)
                capture_factorized_state_adjoint_tile(
                    l, receiver_begin, receiver_count, state_adjoint);

            if (tiled_execution) {
                if (!factorized_tiled_ready)
                    throw std::runtime_error(
                        "Execution R1 tiled coupling workspace exceeds 64 MiB.");

                if (l == 0)
                    Kokkos::parallel_for(
                        "MACEKokkos::cache_factorized_radial_derivatives",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                            execution_space, 0, chunk_edges*embedding),
                        KOKKOS_LAMBDA (const int flat) {
                            const int q = flat%embedding;
                            const int local_edge = flat/embedding;
                            const int edge = chunk_edge_begin+local_edge;
                            const int receiver = edge_receivers(edge);
                            const int type_i = type_map(node_types(receiver));
                            const int type_j = type_map(neigh_types(edge));
                            const int edge_type = type_i <= type_j
                                ? type_i*(2*active_type_count-type_i-1)/2+type_j
                                : type_j*(2*active_type_count-type_j-1)/2+type_i;
                            radial.evaluate_function(
                                edge_type, radial.evaluation_point(r(edge)), q,
                                radial_values(local_edge,q),
                                radial_derivatives(local_edge,q));
                        });

                Kokkos::parallel_for(
                        "MACEKokkos::pack_factorized_state_adjoint",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                            execution_space, 0,
                            receiver_count*embedding*coupling_columns),
                        KOKKOS_LAMBDA (const int flat) {
                            const int column = flat%coupling_columns;
                            const int q = (flat/coupling_columns)%embedding;
                            const int local_receiver =
                                flat/(coupling_columns*embedding);
                            const int channel = column%channels;
                            const int eta_component = column/channels;
                            const int component = eta_component%components;
                            const int eta = eta_component/components;
                            packed_state_adjoint(local_receiver,q,column) =
                                state_adjoint(
                                    local_receiver*components+component,
                                    (eta*embedding+q)*channels+channel);
                        });

                if (factorized_observer_enabled)
                    capture_factorized_state_adjoint_tile(
                        l, receiver_begin, receiver_count,
                        packed_state_adjoint);

                bool uniform_degree = receiver_count > 0;
                const int tile_degree = uniform_degree
                    ? execution_schedule_num_neigh_host[receiver_begin] : 0;
                for (int local_receiver=1;
                     local_receiver<receiver_count; ++local_receiver)
                    uniform_degree = uniform_degree
                        && execution_schedule_num_neigh_host[
                            receiver_begin+local_receiver] == tile_degree;
                {
#if (defined(KOKKOS_ENABLE_CUDA) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)) \
    || (defined(KOKKOS_ENABLE_HIP) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS))
                    if (uniform_degree) {
                        execution_strided_batched_gemm(
                            *factorized_blas_context, factorized_execution_space,
                            radial_values, packed_state_adjoint,
                            coupling_adjoint, receiver_count, tile_degree,
                            embedding, coupling_columns,
                            factorized_max_coupling_columns,
                            coupling_adjoint.extent(1));
                    } else
#endif
                    for (int local_receiver=0;
                         local_receiver<receiver_count; ++local_receiver) {
                        const int receiver = receiver_begin+local_receiver;
                        const int local_edge_begin =
                            execution_receiver_offsets_host[receiver]-chunk_edge_begin;
                        const int local_edge_end = local_edge_begin
                            +execution_schedule_num_neigh_host[receiver];
                        const auto radial_receiver = Kokkos::subview(
                            radial_values,
                            Kokkos::make_pair(local_edge_begin, local_edge_end),
                            Kokkos::ALL);
                        const auto state_receiver = Kokkos::subview(
                            packed_state_adjoint, local_receiver, Kokkos::ALL,
                            Kokkos::make_pair(0, coupling_columns));
                        const auto coupling_receiver = Kokkos::subview(
                            coupling_adjoint,
                            Kokkos::make_pair(local_edge_begin, local_edge_end),
                            Kokkos::make_pair(0, coupling_columns));
                        KokkosBlas::gemm(
                            execution_space, "N", "N", Precision(1),
                            radial_receiver,
                            state_receiver, Precision(0), coupling_receiver);
                    }
                }

                const int source_stride = num_feature_nodes+1;
                const int source_base = chunk*source_stride;
#ifdef KOKKOS_ENABLE_CUDA
                    const auto tiled_source_policy = Kokkos::TeamPolicy<>(
                        execution_space, num_feature_nodes,
                        std::min(channels, 128), 1);
#else
                    const auto tiled_source_policy = Kokkos::TeamPolicy<>(
                        execution_space, num_feature_nodes, Kokkos::AUTO, 1);
#endif
                    Kokkos::parallel_for(
                        "MACEKokkos::reverse_factorized_sources_tiled",
                        tiled_source_policy,
                        KOKKOS_LAMBDA (const TeamMember& member) {
                            const int source = member.league_rank();
                            const int scheduled_begin =
                                source_chunk_offsets(source_base+source);
                            const int scheduled_end =
                                source_chunk_offsets(source_base+source+1);
                            if (scheduled_begin == scheduled_end)
                                return;
                            Kokkos::parallel_for(
                                Kokkos::TeamThreadRange(
                                    member, neighbor_harmonics*channels),
                                [=] (const int flat) {
                                    const int channel = flat%channels;
                                    const int lm2 = flat/channels;
                                    Precision contribution = 0;
                                    const int source_key =
                                        l*neighbor_harmonics+lm2;
                                    for (int scheduled=scheduled_begin;
                                         scheduled<scheduled_end;
                                         ++scheduled) {
                                        const int edge =
                                            source_edges(scheduled);
                                        const int local_edge =
                                            edge-chunk_edge_begin;
                                        for (int term=source_lm_offsets(
                                                 source_key);
                                             term<source_lm_offsets(
                                                 source_key+1); ++term) {
                                            const int column =
                                                source_eta_components(term)
                                                    *channels+channel;
                                            contribution +=
                                                source_coefficients(term)
                                                *coupling_adjoint(
                                                    local_edge,column)
                                                *harmonics_values(
                                                    edge*harmonics
                                                    +source_lm1(term));
                                        }
                                    }
                                    const Precision value = neighbor_adjoint(
                                        source,lm2,channel);
                                    const Precision corrected =
                                        contribution-source_compensation(
                                            source,lm2,channel);
                                    const Precision updated = value+corrected;
                                    source_compensation(source,lm2,channel) =
                                        (updated-value)-corrected;
                                    neighbor_adjoint(source,lm2,channel) =
                                        updated;
                                });
                        });
                Kokkos::parallel_for(
                    "MACEKokkos::reverse_factorized_angular_forces_tiled",
                    Kokkos::TeamPolicy<>(
                        execution_space, chunk_edges, Kokkos::AUTO),
                    KOKKOS_LAMBDA (const TeamMember& member) {
                        const int local_edge = member.league_rank();
                        const int edge = chunk_edge_begin+local_edge;
                        const int neighbor = neigh_indices(edge);
                        Precision f_x = 0;
                        Precision f_y = 0;
                        Precision f_z = 0;
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(
                                member, coupling_columns),
                            [=] (const int column,
                                 Precision& local_f_x,
                                 Precision& local_f_y,
                                 Precision& local_f_z) {
                                const int channel = column%channels;
                                const int eta_component = column/channels;
                                const int component =
                                    eta_component%components;
                                const int eta = eta_component/components;
                                const int lme =
                                    group_lme_begin+component*num_eta+eta;
                                const Precision coupling =
                                    coupling_adjoint(local_edge,column);
                                for (int term=cg_offsets(lme);
                                     term<cg_offsets(lme+1); ++term) {
                                    const int row = cg_rows(term);
                                    const Precision scale =
                                        cg_coefficients(term)*coupling
                                        *neighbor_features(
                                            neighbor,lm2_rows(row),channel);
                                    const int lm1 = lm1_rows(row);
                                    local_f_x += scale*harmonics_gradients(
                                        3*edge*harmonics+lm1);
                                    local_f_y += scale*harmonics_gradients(
                                        (3*edge+1)*harmonics+lm1);
                                    local_f_z += scale*harmonics_gradients(
                                        (3*edge+2)*harmonics+lm1);
                                }
                            }, f_x, f_y, f_z);
                        Kokkos::single(Kokkos::PerTeam(member), [=] () {
                            forces(3*edge) -= static_cast<double>(f_x);
                            forces(3*edge+1) -= static_cast<double>(f_y);
                            forces(3*edge+2) -= static_cast<double>(f_z);
                        });
                    });

                {
#if (defined(KOKKOS_ENABLE_CUDA) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)) \
    || (defined(KOKKOS_ENABLE_HIP) \
        && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS))
                    if (uniform_degree) {
                        execution_strided_batched_gemm(
                            *factorized_blas_context, factorized_execution_space,
                            radial_derivatives, packed_state_adjoint,
                            coupling_adjoint, receiver_count, tile_degree,
                            embedding, coupling_columns,
                            factorized_max_coupling_columns,
                            coupling_adjoint.extent(1));
                    } else
#endif
                    for (int local_receiver=0;
                         local_receiver<receiver_count; ++local_receiver) {
                        const int receiver = receiver_begin+local_receiver;
                        const int local_edge_begin =
                            execution_receiver_offsets_host[receiver]-chunk_edge_begin;
                        const int local_edge_end = local_edge_begin
                            +execution_schedule_num_neigh_host[receiver];
                        const auto radial_receiver = Kokkos::subview(
                            radial_derivatives,
                            Kokkos::make_pair(local_edge_begin, local_edge_end),
                            Kokkos::ALL);
                        const auto state_receiver = Kokkos::subview(
                            packed_state_adjoint, local_receiver, Kokkos::ALL,
                            Kokkos::make_pair(0, coupling_columns));
                        const auto coupling_receiver = Kokkos::subview(
                            coupling_adjoint,
                            Kokkos::make_pair(local_edge_begin, local_edge_end),
                            Kokkos::make_pair(0, coupling_columns));
                        KokkosBlas::gemm(
                            execution_space, "N", "N", Precision(1),
                            radial_receiver,
                            state_receiver, Precision(0), coupling_receiver);
                    }
                }

                Kokkos::parallel_for(
                    "MACEKokkos::reverse_factorized_radial_forces_tiled",
                    Kokkos::TeamPolicy<>(
                        execution_space, chunk_edges, Kokkos::AUTO),
                    KOKKOS_LAMBDA (const TeamMember& member) {
                        const int local_edge = member.league_rank();
                        const int edge = chunk_edge_begin+local_edge;
                        const int neighbor = neigh_indices(edge);
                        Precision radial_force = 0;
                        Kokkos::parallel_reduce(
                            Kokkos::TeamThreadRange(
                                member, coupling_columns),
                            [=] (const int column, Precision& local_force) {
                                const int channel = column%channels;
                                const int eta_component = column/channels;
                                const int component =
                                    eta_component%components;
                                const int eta = eta_component/components;
                                const int lme =
                                    group_lme_begin+component*num_eta+eta;
                                const Precision coupling =
                                    coupling_adjoint(local_edge,column);
                                for (int term=cg_offsets(lme);
                                     term<cg_offsets(lme+1); ++term) {
                                    const int row = cg_rows(term);
                                    local_force += cg_coefficients(term)
                                        *coupling
                                        *neighbor_features(
                                            neighbor,lm2_rows(row),channel)
                                        *harmonics_values(
                                            edge*harmonics+lm1_rows(row));
                                }
                            }, radial_force);
                        Kokkos::single(Kokkos::PerTeam(member), [=] () {
                            const double scale =
                                -static_cast<double>(radial_force);
                            forces(3*edge) += scale*(compact_geometry
                                ? unit_direction(3*edge) : xyz(3*edge)/r(edge));
                            forces(3*edge+1) += scale*(compact_geometry
                                ? unit_direction(3*edge+1)
                                : xyz(3*edge+1)/r(edge));
                            forces(3*edge+2) += scale*(compact_geometry
                                ? unit_direction(3*edge+2)
                                : xyz(3*edge+2)/r(edge));
                        });
                    });
                continue;
            }

            const auto scratch_bytes = admitted_team_scratch_bytes<>(
                "MACEKokkos::reverse_factorized_state",
                {std::size_t(2), static_cast<std::size_t>(embedding),
                 sizeof(Precision)});
            auto policy = Kokkos::TeamPolicy<>(
                execution_space, receiver_count, Kokkos::AUTO, 1);
            policy.set_scratch_size(
                0, Kokkos::PerTeam(scratch_bytes));
            Kokkos::parallel_for(
                "MACEKokkos::reverse_factorized_state", policy,
                KOKKOS_LAMBDA (const TeamMember& member) {
                    const int local_receiver = member.league_rank();
                    const int receiver = receiver_begin+local_receiver;
                    const int type_i = type_map(node_types(receiver));
                    const int edge_begin = first_neigh(receiver);
                    ScratchView radial_data(
                        member.team_scratch(0), 2*embedding);

                    for (int neighbor_offset=0;
                         neighbor_offset<num_neigh(receiver); ++neighbor_offset) {
                        const int edge = edge_begin+neighbor_offset;
                        const int neighbor = neigh_indices(edge);
                        const int type_j = type_map(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        const auto point = radial.evaluation_point(r(edge));
                        Kokkos::parallel_for(
                            Kokkos::TeamVectorRange(member, embedding),
                            [=] (const int q) {
                                radial.evaluate_function(
                                    edge_type, point, q, radial_data(q),
                                    radial_data(embedding+q));
                            });
                        member.team_barrier();

                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(
                                member, num_eta*channels),
                            [=] (const int flat) {
                                const int channel = flat%channels;
                                const int eta = flat/channels;
                                const int path = group_paths(group_begin+eta);
                                for (int row=path_row_offsets(path);
                                     row<path_row_offsets(path+1); ++row) {
                                    Precision radial_dot = 0;
                                    Precision derivative_dot = 0;
                                    for (int term=row_cg_offsets(row);
                                         term<row_cg_offsets(row+1); ++term) {
                                        const int component =
                                            lme_component(row_cg_lme(term));
                                        const int output_row =
                                            local_receiver*components+component;
                                        Precision component_dot = 0;
                                        Precision component_derivative_dot = 0;
                                        for (int q=0; q<embedding; ++q) {
                                            const Precision adjoint = state_adjoint(
                                                output_row,
                                                (eta*embedding+q)*channels+channel);
                                            component_dot +=
                                                adjoint*radial_data(q);
                                            component_derivative_dot += adjoint
                                                *radial_data(embedding+q);
                                        }
                                        radial_dot += row_cg_coefficients(term)
                                            *component_dot;
                                        derivative_dot +=
                                            row_cg_coefficients(term)
                                            *component_derivative_dot;
                                    }
                                    if (radial_dot == Precision(0)
                                        && derivative_dot == Precision(0))
                                        continue;
                                    const int lm1 = lm1_rows(row);
                                    const int lm2 = lm2_rows(row);
                                    const Precision neighbor_feature =
                                        neighbor_features(neighbor,lm2,channel);
                                    const Precision angular =
                                        harmonics_values(edge*harmonics+lm1);
                                    const Precision radial_force =
                                        derivative_dot*angular*neighbor_feature;
                                    const Precision angular_force = radial_dot
                                        *neighbor_feature;
                                    Kokkos::single(
                                        Kokkos::PerThread(member), [=] () {
                                            Kokkos::atomic_add(
                                                &forces(3*edge),
                                                -static_cast<double>(
                                                    radial_force*(compact_geometry
                                                        ? unit_direction(3*edge)
                                                        : xyz(3*edge)/r(edge))
                                                    +angular_force
                                                        *harmonics_gradients(
                                                            3*edge*harmonics+lm1)));
                                            Kokkos::atomic_add(
                                                &forces(3*edge+1),
                                                -static_cast<double>(
                                                    radial_force*(compact_geometry
                                                        ? unit_direction(3*edge+1)
                                                        : xyz(3*edge+1)/r(edge))
                                                    +angular_force
                                                        *harmonics_gradients(
                                                            (3*edge+1)*harmonics
                                                                +lm1)));
                                            Kokkos::atomic_add(
                                                &forces(3*edge+2),
                                                -static_cast<double>(
                                                    radial_force*(compact_geometry
                                                        ? unit_direction(3*edge+2)
                                                        : xyz(3*edge+2)/r(edge))
                                                    +angular_force
                                                        *harmonics_gradients(
                                                            (3*edge+2)*harmonics
                                                                +lm1)));
                                        });
                                }
                            });
                        member.team_barrier();
                    }
                });

            const int source_stride = num_feature_nodes+1;
            const int source_base = chunk*source_stride;
            if (factorized_source_strategy
                == FactorizedSourceStrategy::serial_reference) {
                Kokkos::parallel_for(
                    "MACEKokkos::reverse_factorized_sources_reference",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0, num_feature_nodes),
                    KOKKOS_LAMBDA (const int source) {
                        const int scheduled_begin =
                            source_chunk_offsets(source_base+source);
                        const int scheduled_end =
                            source_chunk_offsets(source_base+source+1);
                        for (int scheduled=scheduled_begin;
                             scheduled<scheduled_end; ++scheduled) {
                            const int edge = source_edges(scheduled);
                            const int receiver = edge_receivers(edge);
                            const int local_receiver = receiver-receiver_begin;
                            const int type_i = type_map(node_types(receiver));
                            const int type_j = type_map(neigh_types(edge));
                            const int edge_type = type_i <= type_j
                                ? type_i*(2*active_type_count-type_i-1)/2+type_j
                                : type_j*(2*active_type_count-type_j-1)/2+type_i;
                            const auto point = radial.evaluation_point(r(edge));
                            for (int channel=0; channel<channels; ++channel)
                                for (int eta=0; eta<num_eta; ++eta) {
                                    const int path = group_paths(group_begin+eta);
                                    for (int row=path_row_offsets(path);
                                         row<path_row_offsets(path+1); ++row) {
                                        Precision radial_dot = 0;
                                        for (int term=row_cg_offsets(row);
                                             term<row_cg_offsets(row+1); ++term) {
                                            const int component = lme_component(
                                                row_cg_lme(term));
                                            const int output_row = local_receiver
                                                *components+component;
                                            Precision component_dot = 0;
                                            for (int q=0; q<embedding; ++q)
                                                component_dot += state_adjoint(
                                                    output_row,
                                                    (eta*embedding+q)*channels+channel)
                                                    *radial.evaluate_function(
                                                        edge_type, point, q);
                                            radial_dot += row_cg_coefficients(term)
                                                *component_dot;
                                        }
                                        const int lm2 = lm2_rows(row);
                                        const Precision contribution = radial_dot
                                            *harmonics_values(
                                                edge*harmonics+lm1_rows(row));
                                        const Precision value = neighbor_adjoint(
                                            source,lm2,channel);
                                        const Precision corrected = contribution
                                            -source_compensation(
                                                source,lm2,channel);
                                        const Precision updated = value+corrected;
                                        source_compensation(source,lm2,channel) =
                                            (updated-value)-corrected;
                                        neighbor_adjoint(source,lm2,channel) =
                                            updated;
                                    }
                                }
                        }
                    });
            } else {
                Kokkos::parallel_for(
                    "MACEKokkos::cache_factorized_radial",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                        execution_space, 0, chunk_edges*embedding),
                    KOKKOS_LAMBDA (const int flat) {
                        const int q = flat%embedding;
                        const int local_edge = flat/embedding;
                        const int edge = chunk_edge_begin+local_edge;
                        const int receiver = edge_receivers(edge);
                        const int type_i = type_map(node_types(receiver));
                        const int type_j = type_map(neigh_types(edge));
                        const int edge_type = type_i <= type_j
                            ? type_i*(2*active_type_count-type_i-1)/2+type_j
                            : type_j*(2*active_type_count-type_j-1)/2+type_i;
                        radial_values(local_edge,q) = radial.evaluate_function(
                            edge_type, radial.evaluation_point(r(edge)), q);
                    });

#ifdef KOKKOS_ENABLE_CUDA
                const auto source_policy = Kokkos::TeamPolicy<>(
                    execution_space, num_feature_nodes,
                    std::min(channels, 128), 1);
#else
                const auto source_policy = Kokkos::TeamPolicy<>(
                    execution_space, num_feature_nodes, Kokkos::AUTO, 1);
#endif
                Kokkos::parallel_for(
                    "MACEKokkos::reverse_factorized_sources_team", source_policy,
                    KOKKOS_LAMBDA (const TeamMember& member) {
                        const int source = member.league_rank();
                        const int scheduled_begin =
                            source_chunk_offsets(source_base+source);
                        const int scheduled_end =
                            source_chunk_offsets(source_base+source+1);
                        if (scheduled_begin == scheduled_end)
                            return;
                        Kokkos::parallel_for(
                            Kokkos::TeamThreadRange(member, channels),
                            [=] (const int channel) {
                                Precision h1_contributions[
                                    streamed_fused_max_num_LM] = {};
                                for (int scheduled=scheduled_begin;
                                     scheduled<scheduled_end; ++scheduled) {
                                    const int edge = source_edges(scheduled);
                                    const int receiver = edge_receivers(edge);
                                    const int local_receiver = receiver-receiver_begin;
                                    const int local_edge = edge-chunk_edge_begin;
                                    for (int eta=0; eta<num_eta; ++eta) {
                                        const int path = group_paths(group_begin+eta);
                                        for (int component=0;
                                             component<components; ++component) {
                                            const int lme = path_component_lme(
                                                path_component_offsets(path)+component);
                                            const int output_row = local_receiver
                                                *components+component;
                                            Precision component_dot = 0;
                                            for (int q=0; q<embedding; ++q)
                                                component_dot += state_adjoint(
                                                    output_row,
                                                    (eta*embedding+q)*channels+channel)
                                                    *radial_values(local_edge,q);
                                            for (int term=cg_offsets(lme);
                                                 term<cg_offsets(lme+1); ++term) {
                                                const int row = cg_rows(term);
                                                h1_contributions[lm2_rows(row)] +=
                                                    cg_coefficients(term)*component_dot
                                                    *harmonics_values(
                                                        edge*harmonics+lm1_rows(row));
                                            }
                                        }
                                    }
                                }
                                for (int lm2=0; lm2<neighbor_harmonics; ++lm2) {
                                    const Precision value = neighbor_adjoint(
                                        source,lm2,channel);
                                    const Precision corrected =
                                        h1_contributions[lm2]-source_compensation(
                                            source,lm2,channel);
                                    const Precision updated = value+corrected;
                                    source_compensation(source,lm2,channel) =
                                        (updated-value)-corrected;
                                    neighbor_adjoint(source,lm2,channel) =
                                        updated;
                                }
                            });
                    });
            }
    }
    finish_factorized_reverse_observation();
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
