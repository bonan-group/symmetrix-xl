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
#include <tuple>
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
template <typename Precision>
void MACEKokkos<Precision>::reduce_node_forces(
    const int num_nodes,
    Kokkos::View<const int*> edge_receivers,
    Kokkos::View<const int*> edge_sources)
{
    if (num_nodes < 0 || edge_receivers.extent(0) != edge_sources.extent(0))
        throw std::invalid_argument("Atom-force reduction extents are inconsistent.");
    if (atom_forces.extent(0) != 3*static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(atom_forces, 3*static_cast<std::size_t>(num_nodes));
    Kokkos::deep_copy(factorized_execution_space, atom_forces, 0.0);
    const auto reduced_forces = atom_forces;
    const auto directed_forces = node_forces;
    const int num_edges = static_cast<int>(edge_sources.extent(0));
    Kokkos::parallel_for(
        "MACEKokkos::reduce_node_forces",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space, 0, num_edges),
        KOKKOS_LAMBDA (const int edge) {
            const int receiver = edge_receivers(edge);
            const int source = edge_sources(edge);
            for (int component=0; component<3; ++component) {
                const double force = directed_forces(3*edge+component);
                Kokkos::atomic_add(
                    &reduced_forces(3*source+component), force);
                Kokkos::atomic_add(
                    &reduced_forces(3*receiver+component), -force);
            }
        });
}

template <typename Precision>
void MACEKokkos<Precision>::reduce_prepared_node_forces(
    const std::uint64_t graph_generation)
{
    if (graph_generation == 0
        || graph_generation != factorized_completed_evaluation_graph_generation
        || graph_generation != factorized_prepared_graph_generation)
        throw std::invalid_argument(
            "Atom-force reduction requires the current completed Execution R1 graph.");
    if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
        if (!single_layer_tiled_outputs_ready)
            throw std::logic_error(
                "Single-layer tiled atom forces are not ready.");
        return;
    }
    reduce_node_forces(
        static_cast<int>(execution_prepared_node_types.extent(0)),
        execution_edge_receivers,
        execution_prepared_neigh_indices);
}

template <typename Precision>
void MACEKokkos<Precision>::reduce_prepared_all_interactions_node_forces(
    const std::uint64_t graph_generation)
{
    if (streamed_edges != MACEStreamedEdgesMode::generic
        || graph_generation == 0
        || graph_generation != all_interactions_prepared_graph_generation
        || graph_generation != all_interactions_completed_graph_generation)
        throw std::invalid_argument(
            "Atom-force reduction requires the current completed streamed all graph.");
    const auto active_sources = Kokkos::subview(
        all_interactions_active_neigh_indices,
        Kokkos::make_pair(
            std::size_t(0),
            static_cast<std::size_t>(all_interactions_active_edge_count)));
    const auto active_receivers = Kokkos::subview(
        all_interactions_active_edge_receivers,
        Kokkos::make_pair(
            std::size_t(0),
            static_cast<std::size_t>(all_interactions_active_edge_count)));
    reduce_node_forces(
        static_cast<int>(all_interactions_candidate_node_types.extent(0)),
        active_receivers,
        active_sources);
}

template <typename Precision>
void MACEKokkos<Precision>::reduce_stress(
    const double volume,
    Kokkos::View<const double*> xyz)
{
    if (!std::isfinite(volume) || !(volume > 0.0) || xyz.extent(0)%3 != 0)
        throw std::invalid_argument("Stress reduction inputs are inconsistent.");
    const std::size_t num_edges = xyz.extent(0)/3;
    if (node_forces.extent(0) < 3*num_edges)
        throw std::invalid_argument(
            "Stress reduction exceeds the evaluated edge extent.");
    if (stress_tensor.extent(0) != 9)
        Kokkos::realloc(stress_tensor, 9);
    const auto reduced_stress = stress_tensor;
    const auto directed_forces = node_forces;
    const double scale = -1.0/volume;
    for (int component=0; component<9; ++component) {
        const int force_component = component/3;
        const int vector_component = component%3;
        Kokkos::parallel_reduce(
            "MACEKokkos::reduce_stress_component",
            Kokkos::RangePolicy<decltype(factorized_execution_space),
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, num_edges),
            KOKKOS_LAMBDA (const std::size_t edge, double& value) {
                value += scale*directed_forces(3*edge+force_component)
                    *xyz(3*edge+vector_component);
            },
            Kokkos::subview(reduced_stress, component));
    }
}

template <typename Precision>
void MACEKokkos<Precision>::reduce_prepared_stress(
    const double volume,
    const std::uint64_t graph_generation)
{
    if (graph_generation == 0
        || graph_generation != factorized_completed_evaluation_graph_generation
        || graph_generation != factorized_prepared_graph_generation)
        throw std::invalid_argument(
            "Stress reduction requires the current completed Execution R1 graph.");
    if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
        if (!single_layer_tiled_outputs_ready
            || (dual_layer_tiled_plan_active
                ? dual_layer_workspace_virial.extent(0)
                : single_layer_workspace_virial.extent(0)) != 9)
            throw std::logic_error(
                "Single-layer tiled virial is not ready.");
        if (!std::isfinite(volume) || !(volume > 0.0))
            throw std::invalid_argument(
                "Single-layer tiled stress requires a positive finite volume.");
        if (stress_tensor.extent(0) != 9)
            Kokkos::realloc(stress_tensor, 9);
        const auto virial = dual_layer_tiled_plan_active
            ? dual_layer_workspace_virial : single_layer_workspace_virial;
        const auto stress = stress_tensor;
        Kokkos::parallel_for(
            "MACEKokkos::scale_single_layer_virial",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, 9),
            KOKKOS_LAMBDA (const int component) {
                stress(component) = virial(component)/volume;
            });
        return;
    }
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64) {
        if (!std::isfinite(volume) || !(volume > 0.0)
            || execution_prepared_unit_direction.extent(0) < 3*num_edges
            || execution_prepared_r.extent(0) < num_edges)
            throw std::invalid_argument(
                "Compact stress reduction inputs are inconsistent.");
        if (node_forces.extent(0) < 3*num_edges)
            throw std::invalid_argument(
                "Stress reduction exceeds the evaluated edge extent.");
        if (stress_tensor.extent(0) != 9)
            Kokkos::realloc(stress_tensor, 9);
        const auto reduced_stress = stress_tensor;
        const auto directed_forces = node_forces;
        const auto unit_direction = execution_prepared_unit_direction;
        const auto radius = execution_prepared_r;
        const double scale = -1.0/volume;
        for (int component=0; component<9; ++component) {
            const int force_component = component/3;
            const int vector_component = component%3;
            Kokkos::parallel_reduce(
                "MACEKokkos::reduce_compact_stress_component",
                Kokkos::RangePolicy<decltype(factorized_execution_space),
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, num_edges),
                KOKKOS_LAMBDA (const std::size_t edge, double& value) {
                    value += scale*directed_forces(3*edge+force_component)
                        *radius(edge)
                        *static_cast<double>(
                            unit_direction(3*edge+vector_component));
                },
                Kokkos::subview(reduced_stress, component));
        }
        return;
    }
    reduce_stress(
        volume,
        Kokkos::subview(
            execution_prepared_xyz,
            Kokkos::make_pair(std::size_t(0), 3*num_edges)));
}

template <typename Precision>
void MACEKokkos<Precision>::reduce_prepared_all_interactions_stress(
    const double volume,
    const std::uint64_t graph_generation)
{
    if (streamed_edges != MACEStreamedEdgesMode::generic
        || graph_generation == 0
        || graph_generation != all_interactions_prepared_graph_generation
        || graph_generation != all_interactions_completed_graph_generation)
        throw std::invalid_argument(
            "Stress reduction requires the current completed streamed all graph.");
    const std::size_t num_edges =
        static_cast<std::size_t>(all_interactions_active_edge_count);
    reduce_stress(
        volume,
        Kokkos::subview(
            all_interactions_active_xyz,
            Kokkos::make_pair(std::size_t(0), 3*num_edges)));
}

template <typename Precision>
void MACEKokkos<Precision>::compute_node_energies_forces(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    const std::uint64_t execution_graph_generation,
    const bool streamed_schedule_prepared)
{
    const auto evaluation_start = std::chrono::steady_clock::now();
    if (execution_graph_generation != 0) {
        if (!mace_uses_prepared_execution(streamed_edges))
            throw std::invalid_argument(
                "An execution graph token requires streamed_edges='direct'.");
        if (execution_graph_generation != factorized_prepared_graph_generation
            || factorized_schedule_dirty)
            throw std::invalid_argument(
                "Execution R1 graph token is stale; prepare the graph again.");
        const std::size_t graph_edges =
            execution_prepared_neigh_indices.extent(0);
        const bool invalid_radius_extent =
            (single_layer_tiled_plan_active || dual_layer_tiled_plan_active)
            ? r.extent(0) < static_cast<std::size_t>(
                dual_layer_tiled_plan_active
                    ? dual_layer_workspace_active_edge_capacity
                    : single_layer_workspace_active_edge_capacity)
            : r.extent(0) != graph_edges;
        if (num_nodes != static_cast<int>(execution_prepared_node_types.extent(0))
            || invalid_radius_extent
            || (!use_compact_edge_geometry()
                && xyz.extent(0) < 3*graph_edges))
            throw std::invalid_argument(
                "Execution R1 prepared coordinates do not match the graph extents.");
        node_types = execution_prepared_node_types;
        num_neigh = execution_prepared_num_neigh;
        neigh_indices = execution_prepared_neigh_indices;
        neigh_types = execution_prepared_neigh_types;
        factorized_prepared_evaluation_count += 1;
        factorized_topology_validation_skip_count += 1;
    } else if (mace_uses_prepared_execution(streamed_edges)) {
        factorized_fallback_evaluation_count += 1;
    }

    validate_graph_cardinality(
        static_cast<std::size_t>(num_nodes),
        execution_graph_generation == 0
            ? static_cast<std::size_t>(num_nodes)
            : static_cast<std::size_t>(execution_prepared_num_feature_nodes),
        neigh_indices.extent(0));

    prepare_mh0_state_policy_views();
    begin_factorized_production_evaluation();
    single_layer_tiled_outputs_ready = false;
    const std::size_t directed_force_extent = 3*r.extent(0);
    if (execution_graph_generation != 0
        && mace_uses_prepared_execution(streamed_edges)) {
        ensure_execution_result_capacity(
            num_nodes, execution_prepared_num_feature_nodes,
            static_cast<int>(r.extent(0)));
    } else {
        if (node_energies_storage.data() != nullptr
            || node_forces_storage.data() != nullptr) {
            Kokkos::fence("Release prepared execution result capacity");
            node_energies = {};
            node_forces = {};
            node_energies_storage = {};
            node_forces_storage = {};
        }
        if (node_energies.size() != num_nodes)
            Kokkos::realloc(node_energies, num_nodes);
        if (node_forces.size() != directed_force_extent)
            Kokkos::realloc(node_forces, directed_force_extent);
    }
    if (mace_uses_prepared_execution(streamed_edges)) {
        Kokkos::deep_copy(factorized_execution_space, node_energies, 0.0);
        Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
        if (single_layer_tiled_plan_active || dual_layer_tiled_plan_active) {
            Kokkos::deep_copy(factorized_execution_space, atom_forces, 0.0);
            Kokkos::deep_copy(
                factorized_execution_space,
                dual_layer_tiled_plan_active
                    ? dual_layer_workspace_virial
                    : single_layer_workspace_virial,
                0.0);
        }
    } else {
        Kokkos::deep_copy(node_energies, 0.0);
        Kokkos::deep_copy(node_forces, 0.0);
    }
    begin_execution_parameter_gradients();

    if (streamed_edges != MACEStreamedEdgesMode::materialized
        && !mace_uses_prepared_execution(streamed_edges)
        && !streamed_schedule_prepared)
        prepare_streamed_edge_schedule(
            num_nodes, static_cast<int>(neigh_indices.extent(0)), num_neigh);
    if (mace_uses_prepared_execution(streamed_edges)
        && execution_graph_generation == 0)
        prepare_factorized_schedule(
            num_nodes, static_cast<int>(neigh_indices.extent(0)),
            num_neigh, neigh_indices);
    if (has_zbl && !single_layer_tiled_plan_active
        && !dual_layer_tiled_plan_active) {
        if (use_compact_edge_geometry())
            zbl.compute_ZBL(
                factorized_execution_space,
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r,
                execution_prepared_unit_direction, node_energies, node_forces);
        else if (use_factorized_async_inference())
            zbl.compute_ZBL(
                factorized_execution_space,
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r, xyz,
                node_energies, node_forces);
        else
            zbl.compute_ZBL(
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, r, xyz, node_energies, node_forces);
    }
    if (mace_uses_prepared_execution(streamed_edges))
        begin_factorized_observation(
            num_nodes, static_cast<int>(neigh_indices.extent(0)));
    if (streamed_edges != MACEStreamedEdgesMode::generic
        && !mace_uses_prepared_execution(streamed_edges))
        compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    if (!single_layer_readout
        && streamed_edges == MACEStreamedEdgesMode::materialized)
        compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    if (!single_layer_tiled_plan_active && !dual_layer_tiled_plan_active)
        compute_Y(xyz, r, use_factorized_async_inference());

    if (dual_layer_tiled_plan_active) {
        compute_dual_layer_tiled(
            num_nodes, node_types, num_neigh, neigh_indices, xyz, r);
        factorized_execution_space.fence(
            "Dual-layer tiled energy and force evaluation");
        factorized_evaluation_fence_count += 1;
        factorized_last_evaluation_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-evaluation_start).count();
        if (execution_graph_generation != 0)
            complete_factorized_prepared_evaluation(execution_graph_generation);
        return;
    }

    if (single_layer_tiled_plan_active) {
        compute_single_layer_tiled(
            num_nodes, node_types, num_neigh, neigh_types, xyz, r);
        factorized_execution_space.fence(
            "Single-layer tiled energy and force evaluation");
        factorized_evaluation_fence_count += 1;
        factorized_last_evaluation_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-evaluation_start).count();
        if (execution_graph_generation != 0)
            complete_factorized_prepared_evaluation(execution_graph_generation);
        return;
    }

    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges))
        compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
    else
        compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    if (mace_uses_prepared_execution(streamed_edges))
        capture_standard_r0_output(num_nodes);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    if (use_m0_module())
        compute_M0_module(num_nodes, node_types);
    else
        compute_M0(num_nodes, node_types);
    compute_H1(num_nodes);
    add_H1_first_residual(num_nodes, node_types, true);

    if (single_layer_readout) {
        compute_M1(num_nodes, node_types);
        compute_H2(num_nodes, node_types);
        compute_readouts(num_nodes, node_types, false, false);
        reverse_H2(num_nodes, node_types, false);
        reverse_M1(num_nodes, node_types);
    } else {
        const bool execution_direct_r1 =
            mace_uses_prepared_execution(streamed_edges)
            && use_factorized_direct_inference();
        if (execution_direct_r1)
            release_factorized_stateful_workspace();
        if (streamed_edges == MACEStreamedEdgesMode::materialized)
            compute_Phi1(num_nodes, num_neigh, neigh_indices);
        else if (!mace_uses_prepared_execution(streamed_edges)
                 || execution_direct_r1) {
            if (execution_direct_r1 && use_factorized_direct_jit_forward())
                compute_Phi1_streamed_jit(
                    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
            else
                compute_Phi1_streamed(
                    num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        }
        if (mace_uses_prepared_execution(streamed_edges)
            && !execution_direct_r1) {
            compute_factorized(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        }
        else if ((!mace_uses_prepared_execution(streamed_edges)
                  || execution_direct_r1)
                 && !use_receiver_local_phi1()
                 && !use_channel_tiled_phi1())
            compute_A1(num_nodes, !use_factorized_async_inference());
        compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
        compute_M1(num_nodes, node_types);
        compute_H2(num_nodes, node_types);

        compute_readouts(num_nodes, node_types, false, false);

        reverse_H2(num_nodes, node_types, false);
        reverse_M1(num_nodes, node_types);
        reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
        if (mace_uses_prepared_execution(streamed_edges)) {
            if (execution_direct_r1) {
                if (!use_receiver_local_phi1() && !use_channel_tiled_phi1())
                    reverse_A1(
                        num_nodes, !use_factorized_async_inference());
                reverse_factorized_direct(
                    num_nodes, node_types, neigh_indices, neigh_types, xyz, r);
            } else {
                reverse_factorized(
                    num_nodes, node_types, num_neigh, neigh_indices,
                    neigh_types, xyz, r);
            }
        } else {
            reverse_A1(num_nodes);
        }
        if (execution_parameter_gradients_enabled) {
            const auto parameter_start = std::chrono::steady_clock::now();
            compute_factorized_parameter_gradients(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
            execution_parameter_gradients_r1_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-parameter_start).count();
        }
        if (streamed_edges == MACEStreamedEdgesMode::materialized)
            reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);
        else if (!mace_uses_prepared_execution(streamed_edges))
            reverse_Phi1_streamed(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
                xyz, r, false, false);
    }

    reverse_H1(num_nodes);
    if (use_m0_module())
        reverse_M0_module(num_nodes, node_types);
    else
        reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (mace_uses_prepared_execution(streamed_edges))
        begin_standard_r0_reverse_observation(
            num_nodes, static_cast<int>(neigh_indices.extent(0)));
    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges))
        reverse_A0_streamed(
            num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    else
        reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (mace_uses_prepared_execution(streamed_edges)) {
        finish_standard_r0_reverse_observation();
        if (execution_parameter_gradients_enabled) {
            const auto parameter_start = std::chrono::steady_clock::now();
            compute_standard_r0_parameter_gradients(
                num_nodes, node_types, num_neigh, neigh_types, r);
            execution_parameter_gradients_r0_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-parameter_start).count();
            const auto density_start = std::chrono::steady_clock::now();
            if (A0_scaled)
                compute_execution_density_parameter_gradients(
                    "A0", num_nodes, node_types, num_neigh, neigh_types, r,
                    A0, A0_adj);
            if (A1_scaled)
                compute_execution_density_parameter_gradients(
                    "A1", num_nodes, node_types, num_neigh, neigh_types, r,
                    A1, A1_adj);
            execution_parameter_gradients_density_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-density_start).count();
            execution_parameter_gradients_ready = true;
        }
        factorized_execution_space.fence("Execution R1 energy and force evaluation");
        factorized_evaluation_fence_count += 1;
        factorized_last_evaluation_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-evaluation_start).count();
    }
    if (execution_graph_generation != 0)
        complete_factorized_prepared_evaluation(execution_graph_generation);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_single_layer_tiled(
    const int num_receivers,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*>,
    Kokkos::View<const double*>,
    Kokkos::View<const double*> r)
{
    if (!single_layer_tiled_plan_active || !single_layer_readout)
        throw std::logic_error(
            "Single-layer tiled execution was entered without an active plan.");
    if (single_layer_workspace_active_capacity <= 0
        || single_layer_workspace_arena.data() == nullptr)
        throw std::logic_error(
            "Single-layer tiled workspace was not prepared with the graph.");

    const auto all_node_types = node_types;
    const auto all_num_neigh = num_neigh;
    const auto all_first_neigh = streamed_first_neigh;
    const auto all_node_energies = node_energies;
    const int capacity = single_layer_workspace_active_capacity;
    std::size_t evaluation_batches = 0;
    for (int receiver_begin=0; receiver_begin<num_receivers;
         receiver_begin += capacity) {
        const int receiver_end = std::min(
            num_receivers, receiver_begin+capacity);
        const int receiver_count = receiver_end-receiver_begin;
        const int edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int edge_end = receiver_end == num_receivers
            ? execution_active_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        const auto receiver_rows = Kokkos::make_pair(
            static_cast<std::size_t>(receiver_begin),
            static_cast<std::size_t>(receiver_end));
        const auto batch_node_types = Kokkos::subview(
            all_node_types, receiver_rows);
        const auto batch_num_neigh = Kokkos::subview(
            all_num_neigh, receiver_rows);
        node_energies = Kokkos::subview(all_node_energies, receiver_rows);
        bind_single_layer_tiled_workspace_batch(receiver_count);
        const int edge_count = edge_end-edge_begin;
        const auto batch_neigh_types = Kokkos::subview(
            single_layer_workspace_neigh_types,
            Kokkos::make_pair(
                std::size_t(0), static_cast<std::size_t>(edge_count)));
        const auto batch_r = Kokkos::subview(
            execution_prepared_r,
            Kokkos::make_pair(
                std::size_t(0), static_cast<std::size_t>(edge_count)));
        const auto batch_first_neigh = streamed_first_neigh;
        node_forces = Kokkos::subview(
            node_forces_storage,
            Kokkos::make_pair(
                std::size_t(0), static_cast<std::size_t>(3)*edge_count));
        single_layer_workspace_xyz = node_forces;
        Kokkos::parallel_scan(
            "MACEKokkos::single_layer_tiled_receiver_offsets",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, receiver_count),
            KOKKOS_LAMBDA (
                    const int receiver, int& offset, const bool final) {
                if (final)
                    batch_first_neigh(receiver) = offset;
                offset += batch_num_neigh(receiver);
            });
        prepare_single_layer_tiled_geometry(
            receiver_begin, receiver_count, edge_begin, edge_count);
        const auto batch_xyz = single_layer_workspace_xyz;
        Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
        if (has_zbl) {
            if (use_compact_edge_geometry())
                zbl.compute_ZBL(
                    factorized_execution_space,
                    receiver_count, batch_node_types, batch_num_neigh,
                    batch_neigh_types, atomic_numbers, streamed_first_neigh,
                    batch_r,
                    execution_prepared_unit_direction, node_energies,
                    node_forces, 0);
            else
                zbl.compute_ZBL(
                    factorized_execution_space,
                    receiver_count, batch_node_types, batch_num_neigh,
                    batch_neigh_types, atomic_numbers, streamed_first_neigh,
                    batch_r, batch_xyz, node_energies, node_forces, 0);
        }
        compute_Y(
            batch_xyz, batch_r, use_factorized_async_inference(), 0, edge_count);

        compute_A0_streamed(
            receiver_count, batch_node_types, batch_num_neigh,
            batch_neigh_types, batch_r, 0);
        compute_A0_scaled(
            receiver_count, batch_node_types, batch_num_neigh,
            batch_neigh_types, batch_r);
        if (use_m0_module())
            compute_M0_module(receiver_count, batch_node_types);
        else
            compute_M0(receiver_count, batch_node_types);
        compute_H1(receiver_count);
        add_H1_first_residual(receiver_count, batch_node_types, true);
        compute_M1(receiver_count, batch_node_types);
        compute_H2(receiver_count, batch_node_types);
        compute_readouts(receiver_count, batch_node_types, false, false);
        reverse_H2(receiver_count, batch_node_types, false);
        reverse_M1(receiver_count, batch_node_types);
        reverse_H1(receiver_count);
        if (use_m0_module())
            reverse_M0_module(receiver_count, batch_node_types);
        else
            reverse_M0(receiver_count, batch_node_types);
        reverse_A0_scaled(
            receiver_count, batch_node_types, batch_num_neigh,
            batch_neigh_types, batch_xyz, batch_r);
        reverse_A0_streamed(
            receiver_count, batch_node_types, batch_num_neigh,
            batch_neigh_types, batch_xyz, batch_r,
            receiver_begin, 0, edge_count);
        accumulate_single_layer_tiled_outputs(
            receiver_begin, receiver_count, edge_begin, edge_count);
        evaluation_batches += 1;
    }
    streamed_first_neigh = all_first_neigh;
    node_energies = all_node_energies;
    single_layer_workspace_batches += evaluation_batches;
    single_layer_tiled_evaluations += 1;
    single_layer_tiled_outputs_ready = true;
}

template <typename Precision>
void MACEKokkos<Precision>::compute_dual_layer_tiled(
    const int num_receivers,
    Kokkos::View<const int*>,
    Kokkos::View<const int*>,
    Kokkos::View<const int*>,
    Kokkos::View<const double*>,
    Kokkos::View<const double*>)
{
    if (!dual_layer_tiled_plan_active || single_layer_readout)
        throw std::logic_error(
            "Dual-layer tiled execution was entered without an active plan.");
    if (dual_layer_workspace_active_capacity <= 0
        || dual_layer_workspace_arena.data() == nullptr
        || dual_layer_graph_h1.extent_int(0) < execution_prepared_num_feature_nodes
        || dual_layer_graph_h1_adjoint.extent_int(0)
            < execution_prepared_num_feature_nodes)
        throw std::logic_error(
            "Dual-layer tiled workspace was not prepared with the graph.");
    Kokkos::deep_copy(
        factorized_execution_space, dual_layer_graph_h1_adjoint, Precision(0));
    compute_dual_layer_tiled_phase1(num_receivers);
    compute_dual_layer_tiled_phase2(num_receivers);
    compute_dual_layer_tiled_phase3(num_receivers);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_dual_layer_tiled_phase1(
    const int num_receivers)
{
    compute_dual_layer_tiled_phase(num_receivers, 1);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_dual_layer_tiled_phase2(
    const int num_receivers)
{
    compute_dual_layer_tiled_phase(num_receivers, 2);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_dual_layer_tiled_phase3(
    const int num_receivers)
{
    compute_dual_layer_tiled_phase(num_receivers, 3);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_dual_layer_tiled_phase(
    const int num_receivers, const int phase)
{
    if (!dual_layer_tiled_plan_active || single_layer_readout
        || phase < 1 || phase > 3)
        throw std::logic_error("Invalid dual-layer tiled evaluation phase.");
    const auto all_node_types =
        Kokkos::View<const int*>(execution_prepared_node_types);
    const auto all_num_neigh =
        Kokkos::View<const int*>(execution_prepared_num_neigh);
    const auto all_node_energies = node_energies_storage;
    const auto receiver_features = execution_prepared_receiver_feature_indices;
    const bool identity_receiver_features = receiver_features.extent(0) == 0;
    const int capacity = dual_layer_workspace_active_capacity;
    const int num_tiles = num_receivers/capacity
        +(num_receivers%capacity != 0);
    std::size_t evaluation_batches = 0;

    const auto tile_bounds = [&] (const int tile) {
        const int receiver_begin = tile*capacity;
        const int receiver_end = std::min(
            num_receivers, receiver_begin+capacity);
        const int edge_begin = execution_receiver_offsets_host.at(
            static_cast<std::size_t>(receiver_begin));
        const int edge_end = receiver_end == num_receivers
            ? execution_active_edges
            : execution_receiver_offsets_host.at(
                static_cast<std::size_t>(receiver_end));
        return std::array<int,4>{
            receiver_begin, receiver_end, edge_begin, edge_end};
    };
    const auto prepare_tile = [&] (
        const int receiver_begin,
        const int receiver_end,
        const int edge_begin,
        const int edge_end) {
        const int receiver_count = receiver_end-receiver_begin;
        const int edge_count = edge_end-edge_begin;
        const auto receiver_rows = Kokkos::make_pair(
            static_cast<std::size_t>(receiver_begin),
            static_cast<std::size_t>(receiver_end));
        const auto local_edge_rows = Kokkos::make_pair(
            std::size_t(0), static_cast<std::size_t>(edge_count));
        const auto batch_node_types = Kokkos::subview(
            all_node_types, receiver_rows);
        const auto batch_num_neigh = Kokkos::subview(
            all_num_neigh, receiver_rows);
        const auto batch_neigh_indices = Kokkos::subview(
            execution_prepared_neigh_indices,
            Kokkos::make_pair(
                static_cast<std::size_t>(edge_begin),
                static_cast<std::size_t>(edge_end)));
        const auto batch_neigh_types = Kokkos::subview(
            dual_layer_workspace_neigh_types, local_edge_rows);
        const auto batch_r = Kokkos::subview(
            execution_prepared_r, local_edge_rows);
        node_energies = Kokkos::subview(all_node_energies, receiver_rows);
        node_forces = Kokkos::subview(
            node_forces_storage,
            Kokkos::make_pair(
                std::size_t(0), static_cast<std::size_t>(3)*edge_count));
        single_layer_workspace_xyz = node_forces;
        const auto first_neigh = streamed_first_neigh;
        Kokkos::parallel_scan(
            "MACEKokkos::dual_layer_tiled_receiver_offsets",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, receiver_count),
            KOKKOS_LAMBDA (
                    const int receiver, int& offset, const bool final) {
                if (final)
                    first_neigh(receiver) = offset;
                offset += batch_num_neigh(receiver);
            });
        prepare_single_layer_tiled_geometry(
            receiver_begin, receiver_count, edge_begin, edge_count);
        return std::tuple{
            batch_node_types, batch_num_neigh, batch_neigh_indices,
            batch_neigh_types, batch_r};
    };

    for (int tile=0; tile<num_tiles; ++tile) {
        const auto [receiver_begin, receiver_end, edge_begin, edge_end] =
            tile_bounds(tile);
        const int receiver_count = receiver_end-receiver_begin;
        const int edge_count = edge_end-edge_begin;
        if (phase == 1)
            bind_dual_layer_phase1_workspace(receiver_begin, receiver_count);
        else if (phase == 2)
            bind_dual_layer_phase2_workspace(receiver_begin, receiver_count);
        else
            bind_dual_layer_phase3_workspace(receiver_begin, receiver_count);
        auto [batch_node_types, batch_num_neigh, batch_neigh_indices,
              batch_neigh_types, batch_r] = prepare_tile(
                  receiver_begin, receiver_end, edge_begin, edge_end);
        const auto batch_xyz = single_layer_workspace_xyz;
        Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
        if (phase == 1) {
            if (has_zbl)
                zbl.compute_ZBL(
                    factorized_execution_space,
                    receiver_count, batch_node_types, batch_num_neigh,
                    batch_neigh_types, atomic_numbers, streamed_first_neigh,
                    batch_r, execution_prepared_unit_direction, node_energies,
                    node_forces, 0);
            compute_Y(batch_xyz, batch_r, true, 0, edge_count);
            compute_A0_streamed(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_r, 0);
            compute_A0_scaled(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_r);
            if (use_m0_module())
                compute_M0_module(receiver_count, batch_node_types);
            else
                compute_M0(receiver_count, batch_node_types);
            compute_H1(receiver_count);
            add_H1_first_residual(receiver_count, batch_node_types, true);
            compute_readout_1(receiver_count, batch_node_types, true);
            const auto tile_h1 = H1;
            const auto tile_h1_adjoint = H1_adj;
            const auto graph_h1 = dual_layer_graph_h1;
            const auto graph_h1_adjoint = dual_layer_graph_h1_adjoint;
            Kokkos::parallel_for(
                "MACEKokkos::scatter_dual_layer_phase1_state",
                Kokkos::MDRangePolicy<decltype(factorized_execution_space),
                    Kokkos::Rank<3>>(
                    factorized_execution_space, {0,0,0},
                    {receiver_count,num_LM,num_channels}),
                KOKKOS_LAMBDA (const int receiver, const int LM, const int k) {
                    const int global_receiver = receiver_begin+receiver;
                    const int feature = identity_receiver_features
                        ? global_receiver : receiver_features(global_receiver);
                    graph_h1(feature,LM,k) = tile_h1(receiver,LM,k);
                    graph_h1_adjoint(feature,LM,k) =
                        tile_h1_adjoint(receiver,LM,k);
                });
        } else if (phase == 2) {
            const auto tile_h1 = H1;
            const auto tile_h1_adjoint = H1_adj;
            const auto graph_h1 = dual_layer_graph_h1;
            const auto graph_h1_adjoint = dual_layer_graph_h1_adjoint;
            Kokkos::parallel_for(
                "MACEKokkos::gather_dual_layer_phase2_state",
                Kokkos::MDRangePolicy<decltype(factorized_execution_space),
                    Kokkos::Rank<3>>(
                    factorized_execution_space, {0,0,0},
                    {receiver_count,num_LM,num_channels}),
                KOKKOS_LAMBDA (const int receiver, const int LM, const int k) {
                    const int global_receiver = receiver_begin+receiver;
                    const int feature = identity_receiver_features
                        ? global_receiver : receiver_features(global_receiver);
                    tile_h1(receiver,LM,k) = graph_h1(feature,LM,k);
                    tile_h1_adjoint(receiver,LM,k) =
                        graph_h1_adjoint(feature,LM,k);
                });
            compute_Y(batch_xyz, batch_r, true, 0, edge_count);
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            compute_Phi1_streamed_jit(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_indices, batch_neigh_types, batch_r);
            H1 = tile_h1;
            H1_adj = tile_h1_adjoint;
            compute_A1_scaled(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_r);
            compute_M1(receiver_count, batch_node_types);
            compute_H2(receiver_count, batch_node_types);
            compute_readout_2(receiver_count);
            reverse_H2(receiver_count, batch_node_types, false);
            reverse_M1(receiver_count, batch_node_types);
            reverse_A1_scaled(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_xyz, batch_r);
            Kokkos::parallel_for(
                "MACEKokkos::scatter_dual_layer_phase2_receiver_adjoint",
                Kokkos::MDRangePolicy<decltype(factorized_execution_space),
                    Kokkos::Rank<3>>(
                    factorized_execution_space, {0,0,0},
                    {receiver_count,num_LM,num_channels}),
                KOKKOS_LAMBDA (const int receiver, const int LM, const int k) {
                    const int global_receiver = receiver_begin+receiver;
                    const int feature = identity_receiver_features
                        ? global_receiver : receiver_features(global_receiver);
                    graph_h1_adjoint(feature,LM,k) =
                        tile_h1_adjoint(receiver,LM,k);
                });
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            dual_layer_active_tile = tile;
            dual_layer_active_segment_begin =
                dual_layer_tile_segment_offsets_host.at(
                    static_cast<std::size_t>(tile));
            dual_layer_active_segment_count =
                dual_layer_tile_segment_offsets_host.at(
                    static_cast<std::size_t>(tile+1))
                -dual_layer_active_segment_begin;
            reverse_factorized_direct(
                receiver_count, batch_node_types, batch_neigh_indices,
                batch_neigh_types, batch_xyz, batch_r);
        } else {
            const auto tile_h1_adjoint = H1_adj;
            const auto graph_h1_adjoint = dual_layer_graph_h1_adjoint;
            Kokkos::parallel_for(
                "MACEKokkos::gather_dual_layer_phase3_adjoint",
                Kokkos::MDRangePolicy<decltype(factorized_execution_space),
                    Kokkos::Rank<3>>(
                    factorized_execution_space, {0,0,0},
                    {receiver_count,num_LM,num_channels}),
                KOKKOS_LAMBDA (const int receiver, const int LM, const int k) {
                    const int global_receiver = receiver_begin+receiver;
                    const int feature = identity_receiver_features
                        ? global_receiver : receiver_features(global_receiver);
                    tile_h1_adjoint(receiver,LM,k) =
                        graph_h1_adjoint(feature,LM,k);
                });
            compute_Y(batch_xyz, batch_r, true, 0, edge_count);
            compute_A0_streamed(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_r, 0);
            compute_A0_scaled(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_r);
            reverse_H1(receiver_count);
            if (use_m0_module())
                reverse_M0_module(receiver_count, batch_node_types);
            else
                reverse_M0(receiver_count, batch_node_types);
            reverse_A0_scaled(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_xyz, batch_r);
            reverse_A0_streamed(
                receiver_count, batch_node_types, batch_num_neigh,
                batch_neigh_types, batch_xyz, batch_r,
                receiver_begin, 0, edge_count);
        }
        accumulate_single_layer_tiled_outputs(
            receiver_begin, receiver_count, edge_begin, edge_count);
        evaluation_batches += 1;
    }

    streamed_first_neigh = {};
    node_energies = all_node_energies;
    H1 = dual_layer_graph_h1;
    H1_adj = dual_layer_graph_h1_adjoint;
    dual_layer_workspace_batches += evaluation_batches;
    if (phase == 3) {
        dual_layer_tiled_evaluations += 1;
        single_layer_tiled_outputs_ready = true;
    }
}

template <typename Precision>
void MACEKokkos<Precision>::accumulate_single_layer_tiled_outputs(
    const int receiver_begin,
    const int receiver_count,
    const int edge_begin,
    const int edge_count)
{
    if (receiver_begin < 0 || receiver_count <= 0
        || receiver_begin+receiver_count
            > static_cast<int>(execution_prepared_node_types.extent(0))
        || edge_begin < 0 || edge_count < 0
        || static_cast<std::size_t>(edge_begin)
                +static_cast<std::size_t>(edge_count)
            > execution_prepared_neigh_indices.extent(0)
        || node_forces.extent(0) < static_cast<std::size_t>(3)*edge_count)
        throw std::invalid_argument(
            "Single-layer tiled output accumulation has invalid edge bounds.");
    const auto sources = execution_prepared_neigh_indices;
    const auto receiver_features =
        execution_prepared_receiver_feature_indices;
    const bool identity_receiver_features = receiver_features.extent(0) == 0;
    const auto num_neigh = execution_prepared_num_neigh;
    const auto first_neigh = streamed_first_neigh;
    const auto directed_forces = node_forces;
    const auto reduced_forces = atom_forces;
    using team_policy = Kokkos::TeamPolicy<
        decltype(factorized_execution_space)>;
    using member_type = typename team_policy::member_type;
    Kokkos::parallel_for(
        "MACEKokkos::accumulate_single_layer_atom_forces",
        team_policy(factorized_execution_space, receiver_count, Kokkos::AUTO),
        KOKKOS_LAMBDA (const member_type& team) {
            const int local_receiver = team.league_rank();
            const int receiver = receiver_begin+local_receiver;
            const int receiver_feature = identity_receiver_features
                ? receiver : receiver_features(receiver);
            const int local_edge_begin = first_neigh(local_receiver);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, num_neigh(receiver)),
                [=] (const int neighbor) {
                    const int local_edge = local_edge_begin+neighbor;
                    const int edge = edge_begin+local_edge;
                    const int source = sources(edge);
                    for (int component=0; component<3; ++component) {
                        const double force =
                            directed_forces(3*local_edge+component);
                        Kokkos::atomic_add(
                            &reduced_forces(3*source+component), force);
                        Kokkos::atomic_add(
                            &reduced_forces(3*receiver_feature+component), -force);
                    }
                });
        });

    if (stress_tensor.extent(0) != 9)
        Kokkos::realloc(stress_tensor, 9);
    const auto batch_virial = stress_tensor;
    const auto virial = dual_layer_tiled_plan_active
        ? dual_layer_workspace_virial : single_layer_workspace_virial;
    const bool compact = use_compact_edge_geometry();
    const auto direction = execution_prepared_unit_direction;
    const auto radius = execution_prepared_r;
    const auto coordinates = execution_prepared_xyz;
    for (int component=0; component<9; ++component) {
        const int force_component = component/3;
        const int vector_component = component%3;
        Kokkos::parallel_reduce(
            "MACEKokkos::reduce_single_layer_batch_virial",
            Kokkos::RangePolicy<decltype(factorized_execution_space)>(
                factorized_execution_space, 0, edge_count),
            KOKKOS_LAMBDA (const int local_edge, double& value) {
                const double displacement = compact
                    ? radius(local_edge)*static_cast<double>(
                        direction(3*local_edge+vector_component))
                    : coordinates(3*local_edge+vector_component);
                value -= directed_forces(3*local_edge+force_component)
                    *displacement;
            },
            Kokkos::subview(batch_virial, component));
    }
    Kokkos::parallel_for(
        "MACEKokkos::accumulate_single_layer_batch_virial",
        Kokkos::RangePolicy<decltype(factorized_execution_space)>(
            factorized_execution_space, 0, 9),
        KOKKOS_LAMBDA (const int component) {
            virial(component) += batch_virial(component);
        });
}

template <typename Precision>
void MACEKokkos<Precision>::compute_node_energies_forces_field(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const int*> num_neigh,
    Kokkos::View<const int*> neigh_indices,
    Kokkos::View<const int*> neigh_types,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    Kokkos::View<const double*> electric_field,
    const std::uint64_t execution_graph_generation)
{
    const auto evaluation_start = std::chrono::steady_clock::now();
    if (!has_field_coupling)
        throw std::invalid_argument("MACEKokkos::compute_node_energies_forces_field requires field coupling.");
    if (execution_graph_generation != 0) {
        if (!mace_uses_prepared_execution(streamed_edges))
            throw std::invalid_argument(
                "An execution graph token requires streamed_edges='direct'.");
        if (execution_graph_generation != factorized_prepared_graph_generation
            || factorized_schedule_dirty)
            throw std::invalid_argument(
                "Execution R1 graph token is stale; prepare the graph again.");
        const std::size_t graph_edges =
            execution_prepared_neigh_indices.extent(0);
        const bool invalid_radius_extent = single_layer_tiled_plan_active
            ? r.extent(0) < static_cast<std::size_t>(
                single_layer_workspace_active_edge_capacity)
            : r.extent(0) != graph_edges;
        if (num_nodes != static_cast<int>(execution_prepared_node_types.extent(0))
            || invalid_radius_extent
            || (!use_compact_edge_geometry()
                && xyz.extent(0) < 3*graph_edges))
            throw std::invalid_argument(
                "Execution R1 prepared coordinates do not match the graph extents.");
        node_types = execution_prepared_node_types;
        num_neigh = execution_prepared_num_neigh;
        neigh_indices = execution_prepared_neigh_indices;
        neigh_types = execution_prepared_neigh_types;
        factorized_prepared_evaluation_count += 1;
        factorized_topology_validation_skip_count += 1;
    } else if (mace_uses_prepared_execution(streamed_edges)) {
        factorized_fallback_evaluation_count += 1;
    }

    validate_graph_cardinality(
        static_cast<std::size_t>(num_nodes),
        execution_graph_generation == 0
            ? static_cast<std::size_t>(num_nodes)
            : static_cast<std::size_t>(execution_prepared_num_feature_nodes),
        neigh_indices.extent(0));

    prepare_mh0_state_policy_views();
    begin_factorized_production_evaluation();
    const std::size_t directed_force_extent = 3*r.extent(0);
    if (execution_graph_generation != 0
        && mace_uses_prepared_execution(streamed_edges)) {
        ensure_execution_result_capacity(
            num_nodes, execution_prepared_num_feature_nodes,
            static_cast<int>(r.extent(0)));
    } else {
        if (node_energies_storage.data() != nullptr
            || node_forces_storage.data() != nullptr) {
            Kokkos::fence("Release prepared execution result capacity");
            node_energies = {};
            node_forces = {};
            node_energies_storage = {};
            node_forces_storage = {};
        }
        if (node_energies.size() != num_nodes)
            Kokkos::realloc(node_energies, num_nodes);
        if (node_forces.size() != directed_force_extent)
            Kokkos::realloc(node_forces, directed_force_extent);
    }
    if (mace_uses_prepared_execution(streamed_edges)) {
        Kokkos::deep_copy(factorized_execution_space, node_energies, 0.0);
        Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
    } else {
        Kokkos::deep_copy(node_energies, 0.0);
        Kokkos::deep_copy(node_forces, 0.0);
    }
    begin_execution_parameter_gradients();

    if (streamed_edges != MACEStreamedEdgesMode::materialized
        && !mace_uses_prepared_execution(streamed_edges))
        prepare_streamed_edge_schedule(
            num_nodes, static_cast<int>(neigh_indices.extent(0)), num_neigh);
    if (mace_uses_prepared_execution(streamed_edges)
        && execution_graph_generation == 0) {
        prepare_factorized_schedule(
            num_nodes, static_cast<int>(neigh_indices.extent(0)),
            num_neigh, neigh_indices);
    }
    if (has_zbl) {
        if (use_compact_edge_geometry())
            zbl.compute_ZBL(
                factorized_execution_space,
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r,
                execution_prepared_unit_direction, node_energies, node_forces);
        else if (use_factorized_async_inference())
            zbl.compute_ZBL(
                factorized_execution_space,
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r, xyz,
                node_energies, node_forces);
        else
            zbl.compute_ZBL(
                num_nodes, node_types, num_neigh, neigh_types,
                atomic_numbers, r, xyz, node_energies, node_forces);
    }
    if (mace_uses_prepared_execution(streamed_edges))
        begin_factorized_observation(
            num_nodes, static_cast<int>(neigh_indices.extent(0)));
    if (streamed_edges != MACEStreamedEdgesMode::generic
        && !mace_uses_prepared_execution(streamed_edges))
        compute_R0(num_nodes, node_types, num_neigh, neigh_types, r);
    if (streamed_edges == MACEStreamedEdgesMode::materialized)
        compute_R1(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_Y(xyz, r, use_factorized_async_inference());

    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges))
        compute_A0_streamed(num_nodes, node_types, num_neigh, neigh_types, r);
    else
        compute_A0(num_nodes, node_types, num_neigh, neigh_types);
    if (mace_uses_prepared_execution(streamed_edges))
        capture_standard_r0_output(num_nodes);
    compute_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    if (use_m0_module())
        compute_M0_module(num_nodes, node_types);
    else
        compute_M0(num_nodes, node_types);
    compute_H1_product(num_nodes);
    add_H1_first_residual(num_nodes, node_types, false);
    compute_field_H1(num_nodes, electric_field);
    compute_H1_linear_up(num_nodes);

    const bool execution_direct_r1 =
        mace_uses_prepared_execution(streamed_edges)
        && use_factorized_direct_inference();
    if (execution_direct_r1)
        release_factorized_stateful_workspace();
    if (streamed_edges == MACEStreamedEdgesMode::materialized)
        compute_Phi1(num_nodes, num_neigh, neigh_indices);
    else if (!mace_uses_prepared_execution(streamed_edges)
             || execution_direct_r1) {
        if (execution_direct_r1 && use_factorized_direct_jit_forward())
            compute_Phi1_streamed_jit(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        else
            compute_Phi1_streamed(
                num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    }
    if (mace_uses_prepared_execution(streamed_edges)
        && !execution_direct_r1) {
        compute_factorized(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
    }
    else if ((!mace_uses_prepared_execution(streamed_edges)
              || execution_direct_r1)
             && !use_receiver_local_phi1()
             && !use_channel_tiled_phi1())
        compute_A1(num_nodes);
    compute_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, r);
    compute_M1(num_nodes, node_types);
    compute_H2(num_nodes, node_types);

    compute_readouts(num_nodes, node_types);

    reverse_H2(num_nodes, node_types, false);
    reverse_M1(num_nodes, node_types);
    reverse_A1_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (mace_uses_prepared_execution(streamed_edges)) {
        if (execution_direct_r1) {
            if (!use_receiver_local_phi1() && !use_channel_tiled_phi1())
                reverse_A1(
                    num_nodes, !use_factorized_direct_jit_reverse());
            reverse_factorized_direct(
                num_nodes, node_types, neigh_indices, neigh_types, xyz, r);
        } else {
            reverse_factorized(
                num_nodes, node_types, num_neigh, neigh_indices,
                neigh_types, xyz, r);
        }
    } else {
        reverse_A1(num_nodes);
    }
    if (execution_parameter_gradients_enabled) {
        const auto parameter_start = std::chrono::steady_clock::now();
        compute_factorized_parameter_gradients(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types, r);
        execution_parameter_gradients_r1_ms =
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-parameter_start).count();
    }
    if (streamed_edges == MACEStreamedEdgesMode::materialized)
        reverse_Phi1(num_nodes, num_neigh, neigh_indices, xyz, r, false, false);
    else if (!mace_uses_prepared_execution(streamed_edges))
        reverse_Phi1_streamed(
            num_nodes, node_types, num_neigh, neigh_indices, neigh_types,
            xyz, r, false, false);

    reverse_H1_linear_up(num_nodes);
    reverse_field_H1(num_nodes, electric_field);
    reverse_H1_product(num_nodes);
    if (use_m0_module())
        reverse_M0_module(num_nodes, node_types);
    else
        reverse_M0(num_nodes, node_types);
    reverse_A0_scaled(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (mace_uses_prepared_execution(streamed_edges))
        begin_standard_r0_reverse_observation(
            num_nodes, static_cast<int>(neigh_indices.extent(0)));
    if (streamed_edges == MACEStreamedEdgesMode::generic
        || mace_uses_prepared_execution(streamed_edges))
        reverse_A0_streamed(
            num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    else
        reverse_A0(num_nodes, node_types, num_neigh, neigh_types, xyz, r);
    if (mace_uses_prepared_execution(streamed_edges)) {
        finish_standard_r0_reverse_observation();
        if (execution_parameter_gradients_enabled) {
            const auto parameter_start = std::chrono::steady_clock::now();
            compute_standard_r0_parameter_gradients(
                num_nodes, node_types, num_neigh, neigh_types, r);
            execution_parameter_gradients_r0_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-parameter_start).count();
            const auto density_start = std::chrono::steady_clock::now();
            if (A0_scaled)
                compute_execution_density_parameter_gradients(
                    "A0", num_nodes, node_types, num_neigh, neigh_types, r,
                    A0, A0_adj);
            if (A1_scaled)
                compute_execution_density_parameter_gradients(
                    "A1", num_nodes, node_types, num_neigh, neigh_types, r,
                    A1, A1_adj);
            execution_parameter_gradients_density_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()-density_start).count();
            execution_parameter_gradients_ready = true;
        }
        factorized_execution_space.fence("Execution R1 field evaluation");
        factorized_evaluation_fence_count += 1;
        factorized_last_evaluation_ms = std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-evaluation_start).count();
    }
    if (execution_graph_generation != 0)
        complete_factorized_prepared_evaluation(execution_graph_generation);
}

template <typename Precision>
void MACEKokkos<Precision>::compute_factorized_single_layer_distributed_evaluation(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    const std::uint64_t execution_graph_generation)
{
    if (!single_layer_readout || has_field_coupling)
        throw std::invalid_argument(
            "Complete distributed evaluation requires a standard single-layer model.");
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        throw std::logic_error(
            "A factorized distributed evaluation is already active.");
    if (execution_graph_generation == 0
        || execution_graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty
        || num_receivers
            != static_cast<int>(execution_prepared_node_types.extent(0))
        || num_feature_nodes != execution_prepared_num_feature_nodes
        || r.extent(0) != execution_prepared_neigh_indices.extent(0)
        || xyz.extent(0) != 3*execution_prepared_neigh_indices.extent(0))
        throw std::invalid_argument(
            "Single-layer distributed graph or geometry extents are inconsistent.");

    if (!single_layer_tiled_plan_active) {
        compute_node_energies_forces(
            num_receivers,
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            xyz, r, execution_graph_generation);
        return;
    }
    if (edge_geometry_policy != EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::logic_error(
            "Single-layer fixed-workspace MPI requires compact edge geometry.");
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_geometry_invalid, 0);
    single_layer_explicit_xyz_device = xyz;
    single_layer_explicit_r_device = r;
    compact_edge_geometry_active = true;
    try {
        compute_node_energies_forces(
            num_receivers,
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            xyz, r, execution_graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        single_layer_explicit_xyz_device = {};
        single_layer_explicit_r_device = {};
        throw;
    }
    compact_edge_geometry_active = false;
    single_layer_explicit_xyz_device = {};
    single_layer_explicit_r_device = {};
    validate_prepared_factorized_positions_geometry();
}

template <typename Precision>
void MACEKokkos<Precision>::
compute_factorized_single_layer_distributed_positions_evaluation(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const double*> positions,
    const std::uint64_t execution_graph_generation)
{
    if (!single_layer_readout || has_field_coupling
        || !single_layer_tiled_plan_active)
        throw std::invalid_argument(
            "Distributed position evaluation requires a tiled standard "
            "single-layer model.");
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        throw std::logic_error(
            "A factorized distributed evaluation is already active.");
    if (execution_graph_generation == 0
        || execution_graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty
        || num_receivers
            != static_cast<int>(execution_prepared_node_types.extent(0))
        || num_feature_nodes != execution_prepared_num_feature_nodes
        || positions.extent(0)
            != std::size_t(3)*static_cast<std::size_t>(num_feature_nodes))
        throw std::invalid_argument(
            "Single-layer distributed graph or position extents are inconsistent.");
    if (edge_geometry_policy != EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::logic_error(
            "Single-layer fixed-workspace MPI requires compact edge geometry.");
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_geometry_invalid, 0);
    const auto workspace_r = Kokkos::subview(
        execution_prepared_r,
        Kokkos::make_pair(
            std::size_t(0), static_cast<std::size_t>(
                single_layer_workspace_active_edge_capacity)));
    single_layer_explicit_feature_positions_device = positions;
    compact_edge_geometry_active = true;
    try {
        compute_node_energies_forces(
            num_receivers,
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const int*>(), Kokkos::View<const int*>(),
            Kokkos::View<const double*>(), workspace_r,
            execution_graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        single_layer_explicit_feature_positions_device = {};
        throw;
    }
    compact_edge_geometry_active = false;
    single_layer_explicit_feature_positions_device = {};
    validate_prepared_factorized_positions_geometry();
}

template <typename Precision>
void MACEKokkos<Precision>::begin_factorized_distributed_positions_evaluation(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const double*> positions,
    const std::uint64_t execution_graph_generation)
{
    if (!dual_layer_tiled_plan_active || single_layer_readout
        || has_field_coupling)
        throw std::invalid_argument(
            "Distributed position prefix requires a tiled standard "
            "dual-layer model.");
    if (positions.extent(0)
        != std::size_t(3)*static_cast<std::size_t>(num_feature_nodes))
        throw std::invalid_argument(
            "Dual-layer distributed position extents are inconsistent.");
    if (edge_geometry_policy != EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::logic_error(
            "Dual-layer fixed-workspace MPI requires compact edge geometry.");
    if (execution_prepared_geometry_invalid.extent(0) != 1) {
        execution_prepared_geometry_invalid = Kokkos::View<int*>(
            "Execution prepared geometry invalid", 1);
        factorized_geometry_state_allocation_count += 1;
    }
    Kokkos::deep_copy(
        factorized_execution_space, execution_prepared_geometry_invalid, 0);
    const auto workspace_r = Kokkos::subview(
        execution_prepared_r,
        Kokkos::make_pair(
            std::size_t(0), static_cast<std::size_t>(
                dual_layer_workspace_active_edge_capacity)));
    single_layer_explicit_feature_positions_device = positions;
    compact_edge_geometry_active = true;
    try {
        begin_factorized_distributed_evaluation(
            num_receivers, num_feature_nodes,
            Kokkos::View<const double*>(), workspace_r,
            execution_graph_generation);
    } catch (...) {
        compact_edge_geometry_active = false;
        single_layer_explicit_feature_positions_device = {};
        throw;
    }
}

template <typename Precision>
void MACEKokkos<Precision>::begin_factorized_distributed_evaluation(
    const int num_receivers,
    const int num_feature_nodes,
    Kokkos::View<const double*> xyz,
    Kokkos::View<const double*> r,
    const std::uint64_t execution_graph_generation)
{
    if (factorized_distributed_phase != FactorizedDistributedPhase::idle)
        throw std::logic_error(
            "A factorized distributed evaluation is already active.");
    if (!mace_uses_prepared_execution(streamed_edges))
        throw std::invalid_argument(
            "Distributed prepared evaluation requires streamed_edges='direct'.");
    if (execution_graph_generation == 0
        || execution_graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Distributed factorized evaluation received a stale graph token.");
    const bool dual_tiled = dual_layer_tiled_plan_active;
    const bool invalid_radius_extent = dual_tiled
        ? r.extent(0) < static_cast<std::size_t>(
            dual_layer_workspace_active_edge_capacity)
        : r.extent(0) != execution_prepared_neigh_indices.extent(0);
    if (num_receivers < 0 || num_feature_nodes < num_receivers
        || num_receivers != static_cast<int>(execution_prepared_node_types.extent(0))
        || num_feature_nodes != execution_prepared_num_feature_nodes
        || num_receivers != execution_schedule_num_nodes
        || num_feature_nodes != execution_schedule_num_feature_nodes
        || invalid_radius_extent
        || (!dual_tiled && !use_compact_edge_geometry()
            && xyz.extent(0) < 3*execution_prepared_neigh_indices.extent(0)))
        throw std::invalid_argument(
            "Distributed factorized graph or geometry extents are inconsistent.");
    if (execution_parameter_gradients_enabled
        && num_feature_nodes != num_receivers)
        throw std::invalid_argument(
            "Distributed factorized parameter gradients are not supported.");
    validate_graph_cardinality(
        static_cast<std::size_t>(num_receivers),
        static_cast<std::size_t>(num_feature_nodes),
        execution_prepared_neigh_indices.extent(0));

    if (dual_tiled) {
        try {
            prepare_mh0_state_policy_views();
            factorized_distributed_start = std::chrono::steady_clock::now();
            begin_factorized_production_evaluation();
            factorized_prepared_evaluation_count += 1;
            factorized_topology_validation_skip_count += 1;
            ensure_execution_result_capacity(
                num_receivers, num_feature_nodes, execution_active_edges);
            Kokkos::deep_copy(
                factorized_execution_space, node_energies, 0.0);
            Kokkos::deep_copy(
                factorized_execution_space, atom_forces, 0.0);
            Kokkos::deep_copy(
                factorized_execution_space, dual_layer_workspace_virial, 0.0);
            Kokkos::deep_copy(
                factorized_execution_space,
                dual_layer_graph_h1_adjoint, Precision(0));
            single_layer_tiled_outputs_ready = false;
            begin_execution_parameter_gradients();
            begin_factorized_observation(num_receivers, execution_active_edges);
            compute_dual_layer_tiled_phase1(num_receivers);
            factorized_execution_space.fence(
                "Dual-layer tiled H1 communication boundary");
            factorized_stage_fence_count += 1;
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            factorized_distributed_num_receivers = num_receivers;
            factorized_distributed_num_feature_nodes = num_feature_nodes;
            factorized_distributed_graph_generation = execution_graph_generation;
            factorized_distributed_xyz = xyz;
            factorized_distributed_r = r;
            factorized_distributed_electric_field = {};
            factorized_distributed_phase =
                FactorizedDistributedPhase::prefix_complete;
            return;
        } catch (...) {
            factorized_distributed_phase = FactorizedDistributedPhase::idle;
            factorized_distributed_num_receivers = 0;
            factorized_distributed_num_feature_nodes = 0;
            factorized_distributed_graph_generation = 0;
            factorized_distributed_xyz = {};
            factorized_distributed_r = {};
            factorized_distributed_electric_field = {};
            throw;
        }
    }

    prepare_mh0_state_policy_views();
    factorized_distributed_start = std::chrono::steady_clock::now();
    begin_factorized_production_evaluation();
    factorized_prepared_evaluation_count += 1;
    factorized_topology_validation_skip_count += 1;

    const auto node_types = Kokkos::View<const int*>(execution_prepared_node_types);
    const auto num_neigh = Kokkos::View<const int*>(execution_prepared_num_neigh);
    const auto neigh_types = Kokkos::View<const int*>(execution_prepared_neigh_types);
    ensure_execution_result_capacity(
        num_receivers, num_feature_nodes, static_cast<int>(r.extent(0)));
    Kokkos::deep_copy(factorized_execution_space, node_energies, 0.0);
    Kokkos::deep_copy(factorized_execution_space, node_forces, 0.0);
    begin_execution_parameter_gradients();

    if (has_zbl) {
        if (use_compact_edge_geometry())
            zbl.compute_ZBL(
                factorized_execution_space,
                num_receivers, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r,
                execution_prepared_unit_direction, node_energies, node_forces);
        else
            zbl.compute_ZBL(
                factorized_execution_space,
                num_receivers, node_types, num_neigh, neigh_types,
                atomic_numbers, streamed_first_neigh, r, xyz,
                node_energies, node_forces);
    }
    begin_factorized_observation(
        num_receivers,
        static_cast<int>(execution_prepared_neigh_indices.extent(0)));
    compute_Y(xyz, r, use_factorized_async_inference());
    compute_A0_streamed(
        num_receivers, node_types, num_neigh, neigh_types, r);
    capture_standard_r0_output(num_receivers);
    compute_A0_scaled(
        num_receivers, node_types, num_neigh, neigh_types, r);
    if (use_m0_module())
        compute_M0_module(num_receivers, node_types);
    else
        compute_M0(num_receivers, node_types);
    const int prefix_capacity = std::max(
        num_receivers, execution_planned_receivers);
    if (factorized_distributed_prefix_h1.extent(0)
            < static_cast<std::size_t>(prefix_capacity)
        || factorized_distributed_prefix_h1.extent(1)
            != static_cast<std::size_t>(num_LM)
        || factorized_distributed_prefix_h1.extent(2)
            != static_cast<std::size_t>(num_channels)) {
        factorized_execution_space.fence(
            "Replace distributed factorized prefix H1 capacity");
        if (H1.data() == factorized_distributed_prefix_h1.data())
            H1 = {};
        factorized_distributed_prefix_h1 = {};
        factorized_distributed_prefix_h1 =
            decltype(factorized_distributed_prefix_h1)(
                Kokkos::view_alloc(
                    "Distributed factorized prefix H1",
                    Kokkos::WithoutInitializing),
                prefix_capacity, num_LM, num_channels);
        factorized_distributed_prefix_h1_allocation_count += 1;
    }
    H1 = factorized_distributed_prefix_h1;
    if (has_field_coupling) {
        compute_H1_product(num_receivers);
        add_H1_first_residual(num_receivers, node_types, false);
    } else {
        compute_H1(num_receivers);
        add_H1_first_residual(num_receivers, node_types, true);
    }

    factorized_execution_space.fence(
        "Distributed factorized H1 communication boundary");
    factorized_stage_fence_count += 1;
    factorized_distributed_num_receivers = num_receivers;
    factorized_distributed_num_feature_nodes = num_feature_nodes;
    factorized_distributed_graph_generation = execution_graph_generation;
    factorized_distributed_xyz = xyz;
    factorized_distributed_r = r;
    factorized_distributed_electric_field = {};
    factorized_distributed_phase =
        FactorizedDistributedPhase::prefix_complete;
}

template <typename Precision>
void MACEKokkos<Precision>::continue_factorized_distributed_evaluation(
    Kokkos::View<Precision***,Kokkos::LayoutRight> communicated_h1,
    Kokkos::View<const double*> electric_field)
{
    if (factorized_distributed_phase
        != FactorizedDistributedPhase::prefix_complete)
        throw std::logic_error(
            "Distributed factorized middle phase requires a completed prefix.");
    if (factorized_distributed_graph_generation == 0
        || factorized_distributed_graph_generation
            != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::logic_error(
            "The distributed factorized graph changed after the prefix phase.");
    const int num_receivers = factorized_distributed_num_receivers;
    const int num_feature_nodes = factorized_distributed_num_feature_nodes;
    if (communicated_h1.extent(0) < static_cast<std::size_t>(num_feature_nodes)
        || communicated_h1.extent(1) != static_cast<std::size_t>(num_LM)
        || communicated_h1.extent(2) != static_cast<std::size_t>(num_channels))
        throw std::invalid_argument(
            "Communicated H1 does not match the distributed graph.");
    if (has_field_coupling) {
        if (electric_field.extent(0) != 3
            && electric_field.extent(0)
                != 3*static_cast<std::size_t>(num_feature_nodes))
            throw std::invalid_argument(
                "Distributed MACEField requires a global or per-feature-node field.");
    } else if (electric_field.extent(0) != 0) {
        throw std::invalid_argument(
            "A field was supplied to a standard MACE evaluation.");
    }

    if (dual_layer_tiled_plan_active) {
        try {
            if (communicated_h1.data() != dual_layer_graph_h1.data()) {
                const auto active_communicated_h1 = Kokkos::subview(
                    communicated_h1,
                    Kokkos::make_pair(
                        std::size_t(0),
                        static_cast<std::size_t>(num_feature_nodes)),
                    Kokkos::ALL, Kokkos::ALL);
                Kokkos::deep_copy(
                    factorized_execution_space,
                    dual_layer_graph_h1, active_communicated_h1);
            }
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            compute_dual_layer_tiled_phase2(num_receivers);
            factorized_execution_space.fence(
                "Dual-layer tiled H1 adjoint communication boundary");
            factorized_stage_fence_count += 1;
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            factorized_distributed_phase =
                FactorizedDistributedPhase::middle_complete;
            return;
        } catch (...) {
            factorized_distributed_phase = FactorizedDistributedPhase::idle;
            factorized_distributed_num_receivers = 0;
            factorized_distributed_num_feature_nodes = 0;
            factorized_distributed_graph_generation = 0;
            factorized_distributed_xyz = {};
            factorized_distributed_r = {};
            factorized_distributed_electric_field = {};
            compact_edge_geometry_active = false;
            single_layer_explicit_feature_positions_device = {};
            throw;
        }
    }

    const auto node_types = Kokkos::View<const int*>(execution_prepared_node_types);
    const auto num_neigh = Kokkos::View<const int*>(execution_prepared_num_neigh);
    const auto neigh_indices =
        Kokkos::View<const int*>(execution_prepared_neigh_indices);
    const auto neigh_types = Kokkos::View<const int*>(execution_prepared_neigh_types);
    const auto xyz = factorized_distributed_xyz;
    const auto r = factorized_distributed_r;
    H1 = communicated_h1;
    factorized_distributed_electric_field = electric_field;

    if (has_field_coupling) {
        compute_field_H1(num_feature_nodes, electric_field);
        compute_H1_linear_up(num_feature_nodes);
    }

    const bool execution_direct_r1 = use_factorized_direct_inference();
    if (execution_direct_r1)
        release_factorized_stateful_workspace();
    if (execution_direct_r1) {
        if (use_factorized_direct_jit_forward())
            compute_Phi1_streamed_jit(
                num_receivers, node_types, num_neigh,
                neigh_indices, neigh_types, r);
        else
            compute_Phi1_streamed(
                num_receivers, node_types, num_neigh,
                neigh_indices, neigh_types, r);
        if (!use_receiver_local_phi1() && !use_channel_tiled_phi1())
            compute_A1(num_receivers, !use_factorized_async_inference());
    } else {
        compute_factorized(
            num_receivers, node_types, num_neigh,
            neigh_indices, neigh_types, r);
    }
    compute_A1_scaled(
        num_receivers, node_types, num_neigh, neigh_types, r);
    compute_M1(num_receivers, node_types);
    compute_H2(num_receivers, node_types);
    compute_readouts(num_receivers, node_types, false, false);

    reverse_H2(num_receivers, node_types, false);
    reverse_M1(num_receivers, node_types);
    reverse_A1_scaled(
        num_receivers, node_types, num_neigh, neigh_types, xyz, r);
    if (execution_direct_r1) {
        if (!use_receiver_local_phi1() && !use_channel_tiled_phi1())
            reverse_A1(
                num_receivers, !use_factorized_direct_jit_reverse());
        reverse_factorized_direct(
            num_receivers, node_types, neigh_indices, neigh_types, xyz, r);
    } else {
        reverse_factorized(
            num_receivers, node_types, num_neigh,
            neigh_indices, neigh_types, xyz, r);
    }
    if (execution_parameter_gradients_enabled) {
        const auto parameter_start = std::chrono::steady_clock::now();
        compute_factorized_parameter_gradients(
            num_receivers, node_types, num_neigh,
            neigh_indices, neigh_types, r);
        execution_parameter_gradients_r1_ms =
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-parameter_start).count();
    }
    if (has_field_coupling) {
        reverse_H1_linear_up(num_feature_nodes);
        reverse_field_H1(num_feature_nodes, electric_field);
    }

    factorized_execution_space.fence(
        "Distributed factorized H1 adjoint communication boundary");
    factorized_stage_fence_count += 1;
    factorized_distributed_phase =
        FactorizedDistributedPhase::middle_complete;
}

template <typename Precision>
void MACEKokkos<Precision>::finish_factorized_distributed_evaluation()
{
    if (factorized_distributed_phase
        != FactorizedDistributedPhase::middle_complete)
        throw std::logic_error(
            "Distributed factorized suffix requires a completed middle phase.");
    if (factorized_distributed_graph_generation == 0
        || factorized_distributed_graph_generation
            != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::logic_error(
            "The distributed factorized graph changed before the suffix phase.");
    const int num_receivers = factorized_distributed_num_receivers;
    const auto node_types = Kokkos::View<const int*>(execution_prepared_node_types);
    const auto num_neigh = Kokkos::View<const int*>(execution_prepared_num_neigh);
    const auto neigh_types = Kokkos::View<const int*>(execution_prepared_neigh_types);
    const auto xyz = factorized_distributed_xyz;
    const auto r = factorized_distributed_r;

    if (dual_layer_tiled_plan_active) {
        try {
            H1 = dual_layer_graph_h1;
            H1_adj = dual_layer_graph_h1_adjoint;
            compute_dual_layer_tiled_phase3(num_receivers);
            factorized_execution_space.fence(
                "Dual-layer tiled energy and force evaluation");
            factorized_evaluation_fence_count += 1;
            factorized_last_evaluation_ms =
                std::chrono::duration<double,std::milli>(
                    std::chrono::steady_clock::now()
                    -factorized_distributed_start).count();
            complete_factorized_prepared_evaluation(
                factorized_distributed_graph_generation);
            factorized_distributed_phase = FactorizedDistributedPhase::idle;
            factorized_distributed_num_receivers = 0;
            factorized_distributed_num_feature_nodes = 0;
            factorized_distributed_graph_generation = 0;
            factorized_distributed_xyz = {};
            factorized_distributed_r = {};
            factorized_distributed_electric_field = {};
            if (compact_edge_geometry_active) {
                compact_edge_geometry_active = false;
                single_layer_explicit_feature_positions_device = {};
                validate_prepared_factorized_positions_geometry();
            }
            return;
        } catch (...) {
            factorized_distributed_phase = FactorizedDistributedPhase::idle;
            factorized_distributed_num_receivers = 0;
            factorized_distributed_num_feature_nodes = 0;
            factorized_distributed_graph_generation = 0;
            factorized_distributed_xyz = {};
            factorized_distributed_r = {};
            factorized_distributed_electric_field = {};
            compact_edge_geometry_active = false;
            single_layer_explicit_feature_positions_device = {};
            throw;
        }
    }

    if (has_field_coupling)
        reverse_H1_product(num_receivers);
    else
        reverse_H1(num_receivers);
    if (use_m0_module())
        reverse_M0_module(num_receivers, node_types);
    else
        reverse_M0(num_receivers, node_types);
    reverse_A0_scaled(
        num_receivers, node_types, num_neigh, neigh_types, xyz, r);
    begin_standard_r0_reverse_observation(
        num_receivers,
        static_cast<int>(execution_prepared_neigh_indices.extent(0)));
    reverse_A0_streamed(
        num_receivers, node_types, num_neigh, neigh_types, xyz, r);
    finish_standard_r0_reverse_observation();

    if (execution_parameter_gradients_enabled) {
        const auto parameter_start = std::chrono::steady_clock::now();
        compute_standard_r0_parameter_gradients(
            num_receivers, node_types, num_neigh, neigh_types, r);
        execution_parameter_gradients_r0_ms =
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-parameter_start).count();
        const auto density_start = std::chrono::steady_clock::now();
        if (A0_scaled)
            compute_execution_density_parameter_gradients(
                "A0", num_receivers, node_types, num_neigh, neigh_types, r,
                A0, A0_adj);
        if (A1_scaled)
            compute_execution_density_parameter_gradients(
                "A1", num_receivers, node_types, num_neigh, neigh_types, r,
                A1, A1_adj);
        execution_parameter_gradients_density_ms =
            std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-density_start).count();
        execution_parameter_gradients_ready = true;
    }

    factorized_execution_space.fence(
        "Distributed factorized energy and force evaluation");
    factorized_evaluation_fence_count += 1;
    factorized_last_evaluation_ms = std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-factorized_distributed_start).count();
    complete_factorized_prepared_evaluation(
        factorized_distributed_graph_generation);
    factorized_distributed_phase = FactorizedDistributedPhase::idle;
    factorized_distributed_num_receivers = 0;
    factorized_distributed_num_feature_nodes = 0;
    factorized_distributed_graph_generation = 0;
    factorized_distributed_xyz = {};
    factorized_distributed_r = {};
    factorized_distributed_electric_field = {};
}


template class MACEKokkos<float>;
template class MACEKokkos<double>;
