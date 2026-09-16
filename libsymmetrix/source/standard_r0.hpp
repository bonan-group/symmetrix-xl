#pragma once

#include <cstddef>
#include <type_traits>

#include <Kokkos_Core.hpp>

#include "spherical_harmonic_device.hpp"
#include "tools_kokkos.hpp"

namespace symmetrix::standard_r0 {

// Frozen compatibility accelerator. New R0 topology coverage belongs in RTC.
inline constexpr bool deprecated_for_new_specializations = true;
inline constexpr char deprecation_notice[] =
    "Built-in R0 topology specialization is deprecated; use RTC for new contracts.";

template<typename Precision, typename CoordinateView, typename RadiusView>
KOKKOS_INLINE_FUNCTION Precision direction_component(
    const CoordinateView& coordinates,
    const RadiusView& radius,
    const std::size_t edge,
    const int component,
    const bool coordinates_are_unit)
{
    const Precision value = static_cast<Precision>(
        coordinates(3*edge+component));
    return coordinates_are_unit
        ? value : value/static_cast<Precision>(radius(edge));
}

inline constexpr char module_id[] = "standard-r0-module-module-v2";
inline constexpr int module_revision = 4;
inline constexpr int channels = 0;
inline constexpr int l_max = 3;
inline constexpr int harmonic_count = 16;
inline constexpr int density_state_bytes_per_node = sizeof(double);
inline constexpr int forward_team_size = 1;
inline constexpr int forward_vector_length = 32;
inline constexpr int forward_host_channel_tile = 8;
inline constexpr int coordinate_reverse_host_channel_tile = 128;
inline constexpr int reverse_team_size = l_max+1;
inline constexpr int reverse_vector_length = 32;
inline constexpr int edge16_team_size = 16;
inline constexpr int edge16_vector_length = 16;
inline constexpr int edge32_team_size = 8;
inline constexpr int edge32_vector_length = 32;
inline constexpr int edge_threads_per_block = 256;
inline constexpr int edge_blocks_per_sm = 4;
inline constexpr bool dynamic_channels = true;
inline constexpr char execution_profile[] = "fixed_weight_coordinate";
inline constexpr char structure_fingerprint[] =
    "sha256:87d40a97a59259e75e7d3f51451a9b4b635349e610415f0aadd7199db238364d";

template <typename ExecutionSpace>
inline constexpr bool host_execution_space = std::is_same_v<
    typename ExecutionSpace::memory_space, Kokkos::HostSpace>;

template <typename ExecutionSpace, typename NodeTypesView,
          typename NumNeighView, typename NeighTypesView,
          typename FirstNeighView, typename RadiusView,
          typename TypeMapView, typename ScaleSpline,
          typename DensityStateView>
void launch_density_prepare(
    const ExecutionSpace& execution_space,
    int num_nodes,
    int active_type_count,
    double cutoff,
    NodeTypesView node_types,
    NumNeighView num_neigh,
    NeighTypesView neigh_types,
    FirstNeighView first_neigh,
    RadiusView radius,
    TypeMapView type_to_active,
    ScaleSpline density_scale,
    DensityStateView density_state)
{
    if constexpr (host_execution_space<ExecutionSpace>) {
        Kokkos::parallel_for(
            "StandardR0::density_prepare_host",
            Kokkos::RangePolicy<ExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const std::size_t receiver) {
                const int receiver_type =
                    type_to_active(node_types(receiver));
                const int edge_begin = first_neigh(receiver);
                double density = 0.0;
                for (int local_edge=0;
                     local_edge<num_neigh(receiver); ++local_edge) {
                    const int edge = edge_begin+local_edge;
                    if (!(radius(edge) < cutoff))
                        continue;
                    const int neighbor_type =
                        type_to_active(neigh_types(edge));
                    const int edge_type = receiver_type <= neighbor_type
                        ? receiver_type
                            *(2*active_type_count-receiver_type-1)/2
                            +neighbor_type
                        : neighbor_type
                            *(2*active_type_count-neighbor_type-1)/2
                            +receiver_type;
                    density += density_scale.evaluate_function(
                        edge_type, radius(edge), 0);
                }
                density_state(receiver) = 1.0/(1.0+density);
            });
        return;
    }
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = typename Policy::member_type;
    Kokkos::parallel_for(
        "StandardR0::density_prepare",
        Policy(execution_space, num_nodes, 1, 32),
        KOKKOS_LAMBDA (const Member& team) {
            const int receiver = team.league_rank();
            const int receiver_type = type_to_active(node_types(receiver));
            const int edge_begin = first_neigh(receiver);
            double density;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(team, num_neigh(receiver)),
                [=] (const int local_edge, double& sum) {
                    const int edge = edge_begin+local_edge;
                    if (!(radius(edge) < cutoff))
                        return;
                    const int neighbor_type =
                        type_to_active(neigh_types(edge));
                    const int edge_type = receiver_type <= neighbor_type
                        ? receiver_type
                            *(2*active_type_count-receiver_type-1)/2
                            +neighbor_type
                        : neighbor_type
                            *(2*active_type_count-neighbor_type-1)/2
                            +receiver_type;
                    sum += density_scale.evaluate_function(
                        edge_type, radius(edge), 0);
                }, density);
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                density_state(receiver) = 1.0/(1.0+density);
            });
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename NumNeighView, typename NeighTypesView,
          typename FirstNeighView, typename RadiusView,
          typename TypeMapView, typename CoefficientsView,
          typename HarmonicsView, typename DensityStateView,
          typename OutputView>
void launch_forward(
    const ExecutionSpace& execution_space,
    int num_nodes,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NumNeighView num_neigh,
    NeighTypesView neigh_types,
    FirstNeighView first_neigh,
    RadiusView radius,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    HarmonicsView harmonics_values,
    bool apply_density_scale,
    DensityStateView density_state,
    OutputView output,
    int workspace_edge_begin = 0)
{
    using Precision = typename OutputView::non_const_value_type;
    const int num_intervals = coefficients.extent_int(1);
    const int channel_count = output.extent_int(2);
    const int harmonic_count_runtime = output.extent_int(1);
    int l_max_runtime = 0;
    while ((l_max_runtime+1)*(l_max_runtime+1) < harmonic_count_runtime)
        ++l_max_runtime;
    if constexpr (host_execution_space<ExecutionSpace>) {
        const int channel_tile_count =
            (channel_count+forward_host_channel_tile-1)
            /forward_host_channel_tile;
        Kokkos::parallel_for(
            "StandardR0::forward_host",
            Kokkos::RangePolicy<ExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                execution_space, 0,
                static_cast<std::size_t>(num_nodes)*channel_tile_count),
            KOKKOS_LAMBDA (const std::size_t owner) {
                const std::size_t receiver = owner/channel_tile_count;
                const int channel_begin =
                    (owner%channel_tile_count)*forward_host_channel_tile;
                const int tile_channels = Kokkos::min(
                    forward_host_channel_tile,channel_count-channel_begin);
                const int receiver_type =
                    type_to_active(node_types(receiver));
                const int edge_begin = first_neigh(receiver);
                const Precision inverse_scale = apply_density_scale
                    ? static_cast<Precision>(density_state(receiver))
                    : Precision(1);
                Precision accumulator[
                    harmonic_count*forward_host_channel_tile] = {};
                for (int local_edge=0;
                     local_edge<num_neigh(receiver); ++local_edge) {
                    const int edge = edge_begin+local_edge;
                    if (!(radius(edge) < cutoff))
                        continue;
                    const std::size_t harmonic_offset = static_cast<std::size_t>(
                        edge-workspace_edge_begin)*harmonic_count_runtime;
                    const int neighbor_type =
                        type_to_active(neigh_types(edge));
                    const int edge_type = receiver_type
                        *active_type_count+neighbor_type;
                    int interval = static_cast<int>(Kokkos::floor(
                        (radius(edge)-spline_min)/spline_h));
                    Precision x = static_cast<Precision>(
                        radius(edge)-spline_min-spline_h*interval);
                    if (interval < 0) {
                        interval = 0;
                        x = Precision(0);
                    } else if (interval >= num_intervals) {
                        interval = num_intervals-1;
                        x = static_cast<Precision>(spline_h);
                    }
                    const Precision xx = x*x;
                    const Precision xxx = xx*x;
                    for (int l=0;l<=l_max_runtime;++l) {
                        const int lm_begin = l*l;
                        const int components = 2*l+1;
                        for (int tile_channel=0;
                             tile_channel<tile_channels;++tile_channel) {
                            const int channel = channel_begin+tile_channel;
                            const int function = l*channel_count+channel;
                            const Precision value = coefficients(
                                edge_type,interval,0,function)
                                +coefficients(edge_type,interval,1,function)*x
                                +coefficients(edge_type,interval,2,function)*xx
                                +coefficients(edge_type,interval,3,function)*xxx;
                            for (int component=0;
                                 component<components;++component) {
                                const int lm = lm_begin+component;
                                accumulator[
                                    lm*forward_host_channel_tile+tile_channel]
                                    += value*harmonics_values(
                                        harmonic_offset+lm);
                            }
                        }
                    }
                }
                for (int lm=0;lm<harmonic_count_runtime;++lm)
                    for (int tile_channel=0;
                         tile_channel<tile_channels;++tile_channel)
                        output(receiver,lm,channel_begin+tile_channel) =
                            accumulator[
                                lm*forward_host_channel_tile+tile_channel]
                            *inverse_scale;
            });
        return;
    }
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = typename Policy::member_type;
    using ScratchView = Kokkos::View<
        Precision**, Kokkos::LayoutRight,
        typename Member::scratch_memory_space,
        Kokkos::MemoryUnmanaged>;
    Policy policy(
        execution_space, num_nodes*(l_max_runtime+1),
        forward_team_size, forward_vector_length);
    const auto scratch_bytes = admitted_team_scratch_bytes<ExecutionSpace>(
        "StandardR0::forward",
        {std::size_t(2)*static_cast<std::size_t>(l_max_runtime)+1,
         static_cast<std::size_t>(channel_count), sizeof(Precision)});
    policy.set_scratch_size(0, Kokkos::PerTeam(scratch_bytes));
    Kokkos::parallel_for(
        "StandardR0::forward",
        policy,
        KOKKOS_LAMBDA (const Member& team) {
            const int receiver = team.league_rank()/(l_max_runtime+1);
            const std::size_t receiver_index = receiver;
            const int l = team.league_rank()%(l_max_runtime+1);
            const int lm_begin = l*l;
            const int components = 2*l+1;
            const int receiver_type = type_to_active(node_types(receiver));
            const int edge_begin = first_neigh(receiver);
            const Precision inverse_scale = apply_density_scale
                ? static_cast<Precision>(density_state(receiver))
                : Precision(1);
            ScratchView accumulator(
                team.team_scratch(0), 2*l_max_runtime+1, channel_count);
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, components*channel_count),
                [=] (const int flat) {
                    accumulator(flat/channel_count,flat%channel_count) =
                        Precision(0);
                });
            team.team_barrier();
            for (int local_edge=0;
                 local_edge<num_neigh(receiver); ++local_edge) {
                const int edge = edge_begin+local_edge;
                if (!(radius(edge) < cutoff))
                    continue;
                const std::size_t harmonic_offset = static_cast<std::size_t>(
                    edge-workspace_edge_begin)*harmonic_count_runtime;
                const int neighbor_type =
                    type_to_active(neigh_types(edge));
                const int edge_type = receiver_type
                    *active_type_count+neighbor_type;
                int interval = static_cast<int>(Kokkos::floor(
                    (radius(edge)-spline_min)/spline_h));
                Precision x = static_cast<Precision>(
                    radius(edge)-spline_min-spline_h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = Precision(0);
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = static_cast<Precision>(spline_h);
                }
                const Precision xx = x*x;
                const Precision xxx = xx*x;
                Kokkos::parallel_for(
                    Kokkos::ThreadVectorRange(team, channel_count),
                    [=] (const int channel) {
                        const int function = l*channel_count+channel;
                        const Precision value = coefficients(
                            edge_type,interval,0,function)
                            +coefficients(edge_type,interval,1,function)*x
                            +coefficients(edge_type,interval,2,function)*xx
                            +coefficients(edge_type,interval,3,function)*xxx;
                        accumulator(0,channel) += value*harmonics_values(
                            harmonic_offset+lm_begin);
                        if (l > 0) {
                            accumulator(1,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+1);
                            accumulator(2,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+2);
                        }
                        if (l > 1) {
                            accumulator(3,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+3);
                            accumulator(4,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+4);
                        }
                        if (l > 2) {
                            accumulator(5,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+5);
                            accumulator(6,channel) += value*harmonics_values(
                                harmonic_offset+lm_begin+6);
                        }
                });
            }
            team.team_barrier();
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(team, components*channel_count),
                [=] (const int flat) {
                    output(
                        receiver_index,lm_begin+flat/channel_count,
                        flat%channel_count) = accumulator(
                            flat/channel_count,flat%channel_count)
                            *inverse_scale;
                });
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename NumNeighView, typename NeighTypesView,
          typename FirstNeighView, typename RadiusView,
          typename TypeMapView, typename CoefficientsView,
          typename HarmonicsView, typename DensityStateView,
          typename OutputView>
void launch_forward_all_l(
    const ExecutionSpace& execution_space,
    int num_nodes,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NumNeighView num_neigh,
    NeighTypesView neigh_types,
    FirstNeighView first_neigh,
    RadiusView radius,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    HarmonicsView harmonics_values,
    bool apply_density_scale,
    DensityStateView density_state,
    OutputView output,
    int workspace_edge_begin = 0)
{
    if constexpr (host_execution_space<ExecutionSpace>) {
        launch_forward(
            execution_space, num_nodes, active_type_count,
            spline_h, spline_min, cutoff, node_types, num_neigh, neigh_types,
            first_neigh, radius, type_to_active, coefficients,
            harmonics_values, apply_density_scale, density_state, output,
            workspace_edge_begin);
        return;
    }
    using Precision = typename OutputView::non_const_value_type;
    const int num_intervals = coefficients.extent_int(1);
    const int channel_count = output.extent_int(2);
    const int harmonic_count_runtime = output.extent_int(1);
    Kokkos::parallel_for(
        "StandardR0::forward_all_l",
        Kokkos::RangePolicy<ExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
                execution_space, 0,
                static_cast<std::size_t>(num_nodes)*channel_count),
        KOKKOS_LAMBDA (const std::size_t owner) {
            const std::size_t receiver = owner/channel_count;
            const int channel = owner%channel_count;
            const int receiver_type = type_to_active(node_types(receiver));
            const int edge_begin = first_neigh(receiver);
            Precision accumulator[harmonic_count] = {};
            for (int local_edge=0;
                 local_edge<num_neigh(receiver); ++local_edge) {
                const int edge = edge_begin+local_edge;
                if (!(radius(edge) < cutoff))
                    continue;
                const std::size_t harmonic_offset = static_cast<std::size_t>(
                    edge-workspace_edge_begin)*harmonic_count_runtime;
                const int neighbor_type =
                    type_to_active(neigh_types(edge));
                const int edge_type = receiver_type
                    *active_type_count+neighbor_type;
                int interval = static_cast<int>(Kokkos::floor(
                    (radius(edge)-spline_min)/spline_h));
                Precision x = static_cast<Precision>(
                    radius(edge)-spline_min-spline_h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = Precision(0);
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = static_cast<Precision>(spline_h);
                }
                const Precision xx = x*x;
                const Precision xxx = xx*x;
                for (int l=0; l<=l_max; ++l) {
                    const int function = l*channel_count+channel;
                    const Precision value = coefficients(
                        edge_type,interval,0,function)
                        +coefficients(edge_type,interval,1,function)*x
                        +coefficients(edge_type,interval,2,function)*xx
                        +coefficients(edge_type,interval,3,function)*xxx;
                    for (int lm=l*l; lm<(l+1)*(l+1); ++lm)
                        accumulator[lm] += value
                            *harmonics_values(harmonic_offset+lm);
                }
            }
            const Precision inverse_scale = apply_density_scale
                ? static_cast<Precision>(density_state(receiver))
                : Precision(1);
            for (int lm=0; lm<harmonic_count_runtime; ++lm)
                output(receiver,lm,channel) =
                    accumulator[lm]*inverse_scale;
        });
}

template <typename ExecutionSpace, typename OutputView,
          typename AdjointView, typename DensityStateView,
          typename PrecomputedDotView>
void launch_reverse_prepare(
    const ExecutionSpace& execution_space,
    int num_nodes,
    OutputView output,
    AdjointView output_adjoint,
    DensityStateView density_state,
    PrecomputedDotView precomputed_dot,
    bool use_precomputed_dot)
{
    using Precision = typename AdjointView::non_const_value_type;
    const int channel_count = output_adjoint.extent_int(2);
    const int harmonic_count_runtime = output_adjoint.extent_int(1);
    const int value_count = harmonic_count_runtime*channel_count;
    if constexpr (host_execution_space<ExecutionSpace>) {
        Kokkos::parallel_for(
            "StandardR0::reverse_prepare_host",
            Kokkos::RangePolicy<ExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const std::size_t receiver) {
                const double inverse_scale = density_state(receiver);
                double output_dot_adjoint = use_precomputed_dot
                    ? precomputed_dot(receiver) : 0.0;
                if (!use_precomputed_dot)
                    for (int flat=0; flat<value_count; ++flat) {
                        const int lm = flat/channel_count;
                        const int channel = flat%channel_count;
                        output_dot_adjoint += static_cast<double>(
                            output_adjoint(receiver,lm,channel))
                            *static_cast<double>(output(receiver,lm,channel));
                    }
                for (int flat=0; flat<value_count; ++flat)
                    output_adjoint(
                        receiver,flat/channel_count,flat%channel_count) *=
                        static_cast<Precision>(inverse_scale);
                density_state(receiver) = output_dot_adjoint*inverse_scale;
            });
        return;
    }
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = typename Policy::member_type;
    Kokkos::parallel_for(
        "StandardR0::reverse_prepare",
        Policy(
            execution_space, num_nodes,
            reverse_team_size, reverse_vector_length),
        KOKKOS_LAMBDA (const Member& team) {
            const int receiver = team.league_rank();
            const std::size_t receiver_index = receiver;
            const double inverse_scale = density_state(receiver);
            double output_dot_adjoint = use_precomputed_dot
                ? precomputed_dot(receiver) : 0.0;
            if (!use_precomputed_dot)
                Kokkos::parallel_reduce(
                    Kokkos::TeamVectorRange(team, value_count),
                    [=] (const int flat, double& sum) {
                        const int lm = flat/channel_count;
                        const int channel = flat%channel_count;
                        sum += static_cast<double>(
                            output_adjoint(receiver_index,lm,channel))
                            *static_cast<double>(output(receiver_index,lm,channel));
                    }, output_dot_adjoint);
            Kokkos::parallel_for(
                Kokkos::TeamVectorRange(team, value_count),
                [=] (const int flat) {
                    output_adjoint(
                        receiver_index,flat/channel_count,
                        flat%channel_count) *=
                        static_cast<Precision>(inverse_scale);
                });
            Kokkos::single(Kokkos::PerTeam(team), [=]() {
                density_state(receiver) = output_dot_adjoint*inverse_scale;
            });
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename NumNeighView, typename NeighTypesView,
          typename FirstNeighView, typename CoordinateView,
          typename RadiusView, typename TypeMapView,
          typename CoefficientsView, typename AdjointView,
          typename HarmonicsView, typename HarmonicsGradientView,
          typename ScaleSpline, typename DensityStateView,
          typename ForceView>
void launch_coordinate_reverse(
    const ExecutionSpace& execution_space,
    int num_nodes,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NumNeighView num_neigh,
    NeighTypesView neigh_types,
    FirstNeighView first_neigh,
    CoordinateView xyz,
    RadiusView radius,
    bool coordinates_are_unit,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    bool apply_density_scale,
    ScaleSpline density_scale,
    DensityStateView density_state,
    AdjointView output_adjoint,
    HarmonicsView harmonics_values,
    HarmonicsGradientView harmonics_gradients,
    ForceView forces,
    int workspace_edge_begin = 0)
{
    using Precision = typename AdjointView::non_const_value_type;
    const int num_intervals = coefficients.extent_int(1);
    const int channel_count = output_adjoint.extent_int(2);
    const int harmonic_count_runtime = output_adjoint.extent_int(1);
    const bool recompute_harmonic_gradients =
        harmonics_gradients.data() == nullptr;
    int l_max_runtime = 0;
    while ((l_max_runtime+1)*(l_max_runtime+1) < harmonic_count_runtime)
        ++l_max_runtime;
    if constexpr (host_execution_space<ExecutionSpace>) {
        Kokkos::parallel_for(
            "StandardR0::coordinate_reverse_host",
            Kokkos::RangePolicy<ExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                execution_space, 0, num_nodes),
            KOKKOS_LAMBDA (const std::size_t receiver) {
                const int receiver_type =
                    type_to_active(node_types(receiver));
                const int edge_begin = first_neigh(receiver);
                const double density_factor = apply_density_scale
                    ? density_state(receiver) : 0.0;
                for (int local_edge=0;
                     local_edge<num_neigh(receiver); ++local_edge) {
                    const int edge = edge_begin+local_edge;
                    if (!(radius(edge) < cutoff))
                        continue;
                    const std::size_t workspace_edge =
                        static_cast<std::size_t>(edge-workspace_edge_begin);
                    const std::size_t coordinate_offset = 3*workspace_edge;
                    const std::size_t harmonic_offset =
                        workspace_edge*harmonic_count_runtime;
                    const std::size_t gradient_offset = 3*harmonic_offset;
                    const int neighbor_type =
                        type_to_active(neigh_types(edge));
                    const int edge_type = receiver_type
                        *active_type_count+neighbor_type;
                    int interval = static_cast<int>(Kokkos::floor(
                        (radius(edge)-spline_min)/spline_h));
                    Precision x = static_cast<Precision>(
                        radius(edge)-spline_min-spline_h*interval);
                    if (interval < 0) {
                        interval = 0;
                        x = Precision(0);
                    } else if (interval >= num_intervals) {
                        interval = num_intervals-1;
                        x = static_cast<Precision>(spline_h);
                    }
                    const Precision xx = x*x;
                    const Precision xxx = xx*x;
                    const Precision x_over_r = direction_component<Precision>(
                        xyz, radius, edge, 0, coordinates_are_unit);
                    const Precision y_over_r = direction_component<Precision>(
                        xyz, radius, edge, 1, coordinates_are_unit);
                    const Precision z_over_r = direction_component<Precision>(
                        xyz, radius, edge, 2, coordinates_are_unit);
                    Precision force_x = Precision(0);
                    Precision force_y = Precision(0);
                    Precision force_z = Precision(0);
                    Precision direct_gradients[3*harmonic_count];
                    if (recompute_harmonic_gradients) {
                        const Precision direction[3] = {
                            x_over_r, y_over_r, z_over_r};
                        normalized_spherical_harmonic_gradients_from_direction<3>(
                            direction, static_cast<Precision>(radius(edge)),
                            direct_gradients);
                    }
                    for (int channel_begin=0; channel_begin<channel_count;
                         channel_begin+=coordinate_reverse_host_channel_tile) {
                        const int tile_channels = Kokkos::min(
                            coordinate_reverse_host_channel_tile,
                            channel_count-channel_begin);
                        Precision lane_force_x[
                            coordinate_reverse_host_channel_tile] = {};
                        Precision lane_force_y[
                            coordinate_reverse_host_channel_tile] = {};
                        Precision lane_force_z[
                            coordinate_reverse_host_channel_tile] = {};
                        for (int l=0; l<l_max_runtime+1; ++l) {
                            const int lm_begin = l*l;
                            const int lm_end = (l+1)*(l+1);
                            Precision values[
                                coordinate_reverse_host_channel_tile];
                            Precision derivatives[
                                coordinate_reverse_host_channel_tile];
                            for (int lane=0; lane<tile_channels; ++lane) {
                                const int channel = channel_begin+lane;
                                const int function = l*channel_count+channel;
                                const Precision c1 = coefficients(
                                    edge_type,interval,1,function);
                                const Precision c2 = coefficients(
                                    edge_type,interval,2,function);
                                const Precision c3 = coefficients(
                                    edge_type,interval,3,function);
                                values[lane] = coefficients(
                                    edge_type,interval,0,function)
                                    +c1*x+c2*xx+c3*xxx;
                                derivatives[lane] = c1+2*c2*x+3*c3*xx;
                            }
                            for (int lm=lm_begin; lm<lm_end; ++lm) {
                                const Precision harmonic = harmonics_values(
                                    harmonic_offset+lm);
                                const Precision gradient_x =
                                    recompute_harmonic_gradients
                                    ? direct_gradients[lm]
                                    : harmonics_gradients(gradient_offset+lm);
                                const Precision gradient_y =
                                    recompute_harmonic_gradients
                                    ? direct_gradients[harmonic_count_runtime+lm]
                                    : harmonics_gradients(
                                        gradient_offset
                                            +harmonic_count_runtime+lm);
                                const Precision gradient_z =
                                    recompute_harmonic_gradients
                                    ? direct_gradients[
                                        2*harmonic_count_runtime+lm]
                                    : harmonics_gradients(
                                        gradient_offset
                                            +2*harmonic_count_runtime+lm);
                                for (int lane=0; lane<tile_channels; ++lane) {
                                    const int channel = channel_begin+lane;
                                    const Precision adjoint =
                                        output_adjoint(receiver,lm,channel);
                                    const Precision radial = derivatives[lane]
                                        *harmonic*adjoint;
                                    const Precision angular =
                                        values[lane]*adjoint;
                                    lane_force_x[lane] +=
                                        radial*x_over_r+angular*gradient_x;
                                    lane_force_y[lane] +=
                                        radial*y_over_r+angular*gradient_y;
                                    lane_force_z[lane] +=
                                        radial*z_over_r+angular*gradient_z;
                                }
                            }
                        }
                        for (int lane=0; lane<tile_channels; ++lane) {
                            force_x += lane_force_x[lane];
                            force_y += lane_force_y[lane];
                            force_z += lane_force_z[lane];
                        }
                    }
                    forces(coordinate_offset) -= force_x;
                    forces(coordinate_offset+1) -= force_y;
                    forces(coordinate_offset+2) -= force_z;
                    if (apply_density_scale) {
                        const int density_edge_type =
                            receiver_type <= neighbor_type
                            ? receiver_type
                                *(2*active_type_count-receiver_type-1)/2
                                +neighbor_type
                            : neighbor_type
                                *(2*active_type_count-neighbor_type-1)/2
                                +receiver_type;
                        double density_value = 0.0;
                        double density_derivative = 0.0;
                        density_scale.evaluate_function(
                            density_edge_type, radius(edge), 0,
                            density_value, density_derivative);
                        const double contribution =
                            density_factor*density_derivative;
                        forces(coordinate_offset) +=
                            contribution*x_over_r;
                        forces(coordinate_offset+1) +=
                            contribution*y_over_r;
                        forces(coordinate_offset+2) +=
                            contribution*z_over_r;
                    }
                }
            });
        return;
    }
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = typename Policy::member_type;
    using ScratchView = Kokkos::View<
        Precision*,typename Member::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Policy policy(
        execution_space, num_nodes,
        reverse_team_size, reverse_vector_length);
    if (recompute_harmonic_gradients)
        policy.set_scratch_size(
            0, Kokkos::PerTeam(3*harmonic_count*sizeof(Precision)));
    Kokkos::parallel_for(
        "StandardR0::coordinate_reverse",
        policy,
        KOKKOS_LAMBDA (const Member& team) {
            ScratchView direct_gradients(
                team.team_scratch(0),
                recompute_harmonic_gradients ? 3*harmonic_count : 0);
            const int receiver = team.league_rank();
            const std::size_t receiver_index = receiver;
            const int receiver_type = type_to_active(node_types(receiver));
            const int edge_begin = first_neigh(receiver);
            const double density_factor = apply_density_scale
                ? density_state(receiver) : 0.0;
            for (int local_edge=0;
                 local_edge<num_neigh(receiver); ++local_edge) {
                const int edge = edge_begin+local_edge;
                if (!(radius(edge) < cutoff))
                    continue;
                const std::size_t workspace_edge =
                    static_cast<std::size_t>(edge-workspace_edge_begin);
                const std::size_t coordinate_offset = 3*workspace_edge;
                const std::size_t harmonic_offset =
                    workspace_edge*harmonic_count_runtime;
                const std::size_t gradient_offset = 3*harmonic_offset;
                const int neighbor_type = type_to_active(neigh_types(edge));
                const int edge_type = receiver_type
                    *active_type_count+neighbor_type;
                int interval = static_cast<int>(Kokkos::floor(
                    (radius(edge)-spline_min)/spline_h));
                Precision x = static_cast<Precision>(
                    radius(edge)-spline_min-spline_h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = Precision(0);
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = static_cast<Precision>(spline_h);
                }
                const Precision xx = x*x;
                const Precision xxx = xx*x;
                const Precision x_over_r = direction_component<Precision>(
                    xyz, radius, edge, 0, coordinates_are_unit);
                const Precision y_over_r = direction_component<Precision>(
                    xyz, radius, edge, 1, coordinates_are_unit);
                const Precision z_over_r = direction_component<Precision>(
                    xyz, radius, edge, 2, coordinates_are_unit);
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    if (!recompute_harmonic_gradients)
                        return;
                    const Precision direction[3] = {
                        x_over_r, y_over_r, z_over_r};
                    normalized_spherical_harmonic_gradients_from_direction<3>(
                        direction, static_cast<Precision>(radius(edge)),
                        direct_gradients.data());
                });
                team.team_barrier();
                Precision force_x = Precision(0);
                Precision force_y = Precision(0);
                Precision force_z = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, l_max_runtime+1),
                    [=] (const int l, Precision& local_x,
                         Precision& local_y, Precision& local_z) {
                        const int lm_begin = l*l;
                        const int lm_end = (l+1)*(l+1);
                        Precision l_x = Precision(0);
                        Precision l_y = Precision(0);
                        Precision l_z = Precision(0);
                        Kokkos::parallel_reduce(
                            Kokkos::ThreadVectorRange(team, channel_count),
                            [=] (const int channel, Precision& channel_x,
                                 Precision& channel_y,
                                 Precision& channel_z) {
                                const int function = l*channel_count+channel;
                                const Precision c1 = coefficients(
                                    edge_type,interval,1,function);
                                const Precision c2 = coefficients(
                                    edge_type,interval,2,function);
                                const Precision c3 = coefficients(
                                    edge_type,interval,3,function);
                                const Precision value = coefficients(
                                    edge_type,interval,0,function)
                                    +c1*x+c2*xx+c3*xxx;
                                const Precision derivative =
                                    c1+2*c2*x+3*c3*xx;
                                for (int lm=lm_begin; lm<lm_end; ++lm) {
                                    const Precision adjoint =
                                        output_adjoint(receiver_index,lm,channel);
                                    const Precision radial = derivative
                                        *harmonics_values(
                                            harmonic_offset+lm)
                                        *adjoint;
                                    const Precision angular = value*adjoint;
                                    channel_x += radial*x_over_r+angular
                                        *(recompute_harmonic_gradients
                                            ? direct_gradients(lm)
                                            : harmonics_gradients(
                                                gradient_offset+lm));
                                    channel_y += radial*y_over_r+angular
                                        *(recompute_harmonic_gradients
                                            ? direct_gradients(
                                                harmonic_count_runtime+lm)
                                            : harmonics_gradients(
                                                gradient_offset
                                                    +harmonic_count_runtime+lm));
                                    channel_z += radial*z_over_r+angular
                                        *(recompute_harmonic_gradients
                                            ? direct_gradients(
                                                2*harmonic_count_runtime+lm)
                                            : harmonics_gradients(
                                                gradient_offset
                                                    +2*harmonic_count_runtime+lm));
                                }
                            }, l_x, l_y, l_z);
                        local_x += l_x;
                        local_y += l_y;
                        local_z += l_z;
                    }, force_x, force_y, force_z);
                double density_value = 0.0;
                double density_derivative = 0.0;
                if (apply_density_scale) {
                    const int density_edge_type =
                        receiver_type <= neighbor_type
                        ? receiver_type
                            *(2*active_type_count-receiver_type-1)/2
                            +neighbor_type
                        : neighbor_type
                            *(2*active_type_count-neighbor_type-1)/2
                            +receiver_type;
                    density_scale.evaluate_function(
                        density_edge_type, radius(edge), 0,
                        density_value, density_derivative);
                }
                Kokkos::single(Kokkos::PerTeam(team), [=]() {
                    forces(coordinate_offset) -= force_x;
                    forces(coordinate_offset+1) -= force_y;
                    forces(coordinate_offset+2) -= force_z;
                    if (apply_density_scale) {
                        const double contribution =
                            density_factor*density_derivative;
                        forces(coordinate_offset) +=
                            contribution*x_over_r;
                        forces(coordinate_offset+1) +=
                            contribution*y_over_r;
                        forces(coordinate_offset+2) +=
                            contribution*z_over_r;
                    }
                });
            }
        });
}

template <int EdgeTeamSize, int EdgeVectorLength,
          typename ExecutionSpace, typename NodeTypesView,
          typename NeighTypesView, typename EdgeReceiversView,
          typename CoordinateView, typename RadiusView,
          typename TypeMapView, typename CoefficientsView,
          typename AdjointView, typename HarmonicsView,
          typename HarmonicsGradientView, typename ScaleSpline,
          typename DensityStateView, typename ForceView>
void launch_coordinate_reverse_edge_owned(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    std::size_t num_edges,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NeighTypesView neigh_types,
    EdgeReceiversView edge_receivers,
    CoordinateView xyz,
    RadiusView radius,
    bool coordinates_are_unit,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    bool apply_density_scale,
    ScaleSpline density_scale,
    DensityStateView density_state,
    AdjointView output_adjoint,
    HarmonicsView harmonics_values,
    HarmonicsGradientView harmonics_gradients,
    ForceView forces)
{
    static_assert(
        EdgeTeamSize*EdgeVectorLength == edge_threads_per_block,
        "standard Execution R0 edge kernels require 256 threads");
    if (persistent_blocks <= 0 || num_edges <= 0)
        return;
    using Precision = typename AdjointView::non_const_value_type;
    const int num_intervals = coefficients.extent_int(1);
    const int channel_count = output_adjoint.extent_int(2);
    const int harmonic_count_runtime = output_adjoint.extent_int(1);
    const bool recompute_harmonic_gradients =
        harmonics_gradients.data() == nullptr;
    int l_max_runtime = 0;
    while ((l_max_runtime+1)*(l_max_runtime+1) < harmonic_count_runtime)
        ++l_max_runtime;
    if constexpr (host_execution_space<ExecutionSpace>) {
        Kokkos::parallel_for(
            "StandardR0::coordinate_reverse_edge_host",
            Kokkos::RangePolicy<ExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                execution_space, 0, num_edges),
            KOKKOS_LAMBDA (const std::size_t edge) {
                if (!(radius(edge) < cutoff))
                    return;
                const std::size_t coordinate_offset = 3*edge;
                const std::size_t harmonic_offset =
                    edge*harmonic_count_runtime;
                const std::size_t gradient_offset = 3*harmonic_offset;
                const int receiver = edge_receivers(edge);
                const std::size_t receiver_index = receiver;
                const int receiver_type =
                    type_to_active(node_types(receiver));
                const int neighbor_type =
                    type_to_active(neigh_types(edge));
                const int edge_type = receiver_type
                    *active_type_count+neighbor_type;
                int interval = static_cast<int>(Kokkos::floor(
                    (radius(edge)-spline_min)/spline_h));
                Precision x = static_cast<Precision>(
                    radius(edge)-spline_min-spline_h*interval);
                if (interval < 0) {
                    interval = 0;
                    x = Precision(0);
                } else if (interval >= num_intervals) {
                    interval = num_intervals-1;
                    x = static_cast<Precision>(spline_h);
                }
                const Precision xx = x*x;
                const Precision xxx = xx*x;
                const Precision x_over_r = direction_component<Precision>(
                    xyz, radius, edge, 0, coordinates_are_unit);
                const Precision y_over_r = direction_component<Precision>(
                    xyz, radius, edge, 1, coordinates_are_unit);
                const Precision z_over_r = direction_component<Precision>(
                    xyz, radius, edge, 2, coordinates_are_unit);
                Precision force_x = Precision(0);
                Precision force_y = Precision(0);
                Precision force_z = Precision(0);
                Precision direct_gradients[3*harmonic_count];
                if (recompute_harmonic_gradients) {
                    const Precision direction[3] = {
                        x_over_r, y_over_r, z_over_r};
                    normalized_spherical_harmonic_gradients_from_direction<3>(
                        direction, static_cast<Precision>(radius(edge)),
                        direct_gradients);
                }
                for (int channel=0; channel<channel_count; ++channel) {
                    for (int l=0; l<l_max_runtime+1; ++l) {
                        const int lm_begin = l*l;
                        const int lm_end = (l+1)*(l+1);
                        const int function = l*channel_count+channel;
                        const Precision c1 = coefficients(
                            edge_type,interval,1,function);
                        const Precision c2 = coefficients(
                            edge_type,interval,2,function);
                        const Precision c3 = coefficients(
                            edge_type,interval,3,function);
                        const Precision value = coefficients(
                            edge_type,interval,0,function)
                            +c1*x+c2*xx+c3*xxx;
                        const Precision derivative =
                            c1+2*c2*x+3*c3*xx;
                        for (int lm=lm_begin; lm<lm_end; ++lm) {
                            const Precision adjoint =
                                output_adjoint(receiver_index,lm,channel);
                            const Precision radial = derivative
                                *harmonics_values(
                                    harmonic_offset+lm)
                                *adjoint;
                            const Precision angular = value*adjoint;
                            force_x += radial*x_over_r+angular
                                *(recompute_harmonic_gradients
                                    ? direct_gradients[lm]
                                    : harmonics_gradients(gradient_offset+lm));
                            force_y += radial*y_over_r+angular
                                *(recompute_harmonic_gradients
                                    ? direct_gradients[harmonic_count_runtime+lm]
                                    : harmonics_gradients(
                                        gradient_offset
                                            +harmonic_count_runtime+lm));
                            force_z += radial*z_over_r+angular
                                *(recompute_harmonic_gradients
                                    ? direct_gradients[
                                        2*harmonic_count_runtime+lm]
                                    : harmonics_gradients(
                                        gradient_offset
                                            +2*harmonic_count_runtime+lm));
                        }
                    }
                }
                forces(coordinate_offset) -= static_cast<double>(force_x);
                forces(coordinate_offset+1) -= static_cast<double>(force_y);
                forces(coordinate_offset+2) -= static_cast<double>(force_z);
                if (apply_density_scale) {
                    const int density_edge_type =
                        receiver_type <= neighbor_type
                        ? receiver_type
                            *(2*active_type_count-receiver_type-1)/2
                            +neighbor_type
                        : neighbor_type
                            *(2*active_type_count-neighbor_type-1)/2
                            +receiver_type;
                    double density_value = 0.0;
                    double density_derivative = 0.0;
                    density_scale.evaluate_function(
                        density_edge_type, radius(edge), 0,
                        density_value, density_derivative);
                    const double contribution = density_state(receiver)
                        *density_derivative;
                    forces(coordinate_offset) +=
                        contribution*x_over_r;
                    forces(coordinate_offset+1) +=
                        contribution*y_over_r;
                    forces(coordinate_offset+2) +=
                        contribution*z_over_r;
                }
            });
        return;
    }
    using Policy = Kokkos::TeamPolicy<ExecutionSpace>;
    using Member = typename Policy::member_type;
    using ScratchView = Kokkos::View<
        Precision*,typename Member::scratch_memory_space,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    Policy policy(
        execution_space, persistent_blocks,
        EdgeTeamSize, EdgeVectorLength);
    if (recompute_harmonic_gradients)
        policy.set_scratch_size(
            0, Kokkos::PerTeam(
                EdgeTeamSize*3*harmonic_count*sizeof(Precision)));
    Kokkos::parallel_for(
        EdgeVectorLength == edge16_vector_length
            ? "StandardR0::coordinate_reverse_edge16"
            : "StandardR0::coordinate_reverse_edge32",
        policy,
        KOKKOS_LAMBDA (const Member& team) {
            ScratchView direct_gradient_storage(
                team.team_scratch(0),
                recompute_harmonic_gradients
                    ? EdgeTeamSize*3*harmonic_count : 0);
            for (std::size_t edge_base=
                     static_cast<std::size_t>(team.league_rank())*EdgeTeamSize;
                 edge_base<num_edges;
                 edge_base+=
                     static_cast<std::size_t>(team.league_size())*EdgeTeamSize) {
                const std::size_t edge = edge_base+team.team_rank();
                const bool edge_exists = edge < num_edges;
                const bool edge_active = edge_exists
                    && radius(edge) < cutoff;
                const std::size_t coordinate_offset =
                    static_cast<std::size_t>(3)*edge;
                const std::size_t harmonic_offset =
                    static_cast<std::size_t>(edge)*harmonic_count_runtime;
                const std::size_t gradient_offset = 3*harmonic_offset;
                int receiver = 0;
                int receiver_type = 0;
                int neighbor_type = 0;
                int edge_type = 0;
                int interval = 0;
                Precision x = Precision(0);
                Precision xx = Precision(0);
                Precision xxx = Precision(0);
                Precision x_over_r = Precision(0);
                Precision y_over_r = Precision(0);
                Precision z_over_r = Precision(0);
                if (edge_active) {
                    receiver = edge_receivers(edge);
                    receiver_type = type_to_active(node_types(receiver));
                    neighbor_type = type_to_active(neigh_types(edge));
                    edge_type = receiver_type
                        *active_type_count+neighbor_type;
                    interval = static_cast<int>(Kokkos::floor(
                        (radius(edge)-spline_min)/spline_h));
                    x = static_cast<Precision>(
                        radius(edge)-spline_min-spline_h*interval);
                    if (interval < 0) {
                        interval = 0;
                        x = Precision(0);
                    } else if (interval >= num_intervals) {
                        interval = num_intervals-1;
                        x = static_cast<Precision>(spline_h);
                    }
                    xx = x*x;
                    xxx = xx*x;
                    x_over_r = direction_component<Precision>(
                        xyz, radius, edge, 0, coordinates_are_unit);
                    y_over_r = direction_component<Precision>(
                        xyz, radius, edge, 1, coordinates_are_unit);
                    z_over_r = direction_component<Precision>(
                        xyz, radius, edge, 2, coordinates_are_unit);
                }
                Precision* direct_gradients = recompute_harmonic_gradients
                    ? direct_gradient_storage.data()
                        +team.team_rank()*3*harmonic_count
                    : nullptr;
                Kokkos::single(Kokkos::PerThread(team), [=]() {
                    if (!edge_active || !recompute_harmonic_gradients)
                        return;
                    const Precision direction[3] = {
                        x_over_r, y_over_r, z_over_r};
                    normalized_spherical_harmonic_gradients_from_direction<3>(
                        direction, static_cast<Precision>(radius(edge)),
                        direct_gradients);
                });
                team.team_barrier();
                const std::size_t receiver_index = receiver;
                Precision force_x = Precision(0);
                Precision force_y = Precision(0);
                Precision force_z = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::ThreadVectorRange(team, channel_count),
                    [=] (const int channel, Precision& local_x,
                         Precision& local_y, Precision& local_z) {
                        if (!edge_active)
                            return;
                        for (int l=0; l<l_max_runtime+1; ++l) {
                            const int lm_begin = l*l;
                            const int lm_end = (l+1)*(l+1);
                            const int function = l*channel_count+channel;
                            const Precision c1 = coefficients(
                                edge_type,interval,1,function);
                            const Precision c2 = coefficients(
                                edge_type,interval,2,function);
                            const Precision c3 = coefficients(
                                edge_type,interval,3,function);
                            const Precision value = coefficients(
                                edge_type,interval,0,function)
                                +c1*x+c2*xx+c3*xxx;
                            const Precision derivative =
                                c1+2*c2*x+3*c3*xx;
                            for (int lm=lm_begin; lm<lm_end; ++lm) {
                                const Precision adjoint =
                                    output_adjoint(receiver_index,lm,channel);
                                const Precision radial = derivative
                                    *harmonics_values(
                                        harmonic_offset+lm)
                                    *adjoint;
                                const Precision angular = value*adjoint;
                                local_x += radial*x_over_r+angular
                                    *(recompute_harmonic_gradients
                                        ? direct_gradients[lm]
                                        : harmonics_gradients(
                                            gradient_offset+lm));
                                local_y += radial*y_over_r+angular
                                    *(recompute_harmonic_gradients
                                        ? direct_gradients[
                                            harmonic_count_runtime+lm]
                                        : harmonics_gradients(
                                            gradient_offset
                                                +harmonic_count_runtime+lm));
                                local_z += radial*z_over_r+angular
                                    *(recompute_harmonic_gradients
                                        ? direct_gradients[
                                            2*harmonic_count_runtime+lm]
                                        : harmonics_gradients(
                                            gradient_offset
                                                +2*harmonic_count_runtime+lm));
                            }
                        }
                    }, force_x, force_y, force_z);
                Kokkos::single(Kokkos::PerThread(team), [=]() {
                    if (!edge_active)
                        return;
                    forces(coordinate_offset) -= static_cast<double>(force_x);
                    forces(coordinate_offset+1) -= static_cast<double>(force_y);
                    forces(coordinate_offset+2) -= static_cast<double>(force_z);
                    if (apply_density_scale) {
                        const int density_edge_type =
                            receiver_type <= neighbor_type
                            ? receiver_type
                                *(2*active_type_count-receiver_type-1)/2
                                +neighbor_type
                            : neighbor_type
                                *(2*active_type_count-neighbor_type-1)/2
                                +receiver_type;
                        double density_value = 0.0;
                        double density_derivative = 0.0;
                        density_scale.evaluate_function(
                            density_edge_type, radius(edge), 0,
                            density_value, density_derivative);
                        const double contribution = density_state(receiver)
                            *density_derivative;
                        forces(coordinate_offset) +=
                            contribution*x_over_r;
                        forces(coordinate_offset+1) +=
                            contribution*y_over_r;
                        forces(coordinate_offset+2) +=
                            contribution*z_over_r;
                    }
                });
            }
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename NeighTypesView, typename EdgeReceiversView,
          typename CoordinateView, typename RadiusView,
          typename TypeMapView, typename CoefficientsView,
          typename AdjointView, typename HarmonicsView,
          typename HarmonicsGradientView, typename ScaleSpline,
          typename DensityStateView, typename ForceView>
void launch_coordinate_reverse_edge16(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    std::size_t num_edges,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NeighTypesView neigh_types,
    EdgeReceiversView edge_receivers,
    CoordinateView xyz,
    RadiusView radius,
    bool coordinates_are_unit,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    bool apply_density_scale,
    ScaleSpline density_scale,
    DensityStateView density_state,
    AdjointView output_adjoint,
    HarmonicsView harmonics_values,
    HarmonicsGradientView harmonics_gradients,
    ForceView forces)
{
    launch_coordinate_reverse_edge_owned<
        edge16_team_size, edge16_vector_length>(
            execution_space, persistent_blocks, num_edges, active_type_count,
            spline_h, spline_min, cutoff, node_types, neigh_types, edge_receivers,
            xyz, radius, coordinates_are_unit, type_to_active, coefficients,
            apply_density_scale,
            density_scale, density_state, output_adjoint, harmonics_values,
            harmonics_gradients, forces);
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename NeighTypesView, typename EdgeReceiversView,
          typename CoordinateView, typename RadiusView,
          typename TypeMapView, typename CoefficientsView,
          typename AdjointView, typename HarmonicsView,
          typename HarmonicsGradientView, typename ScaleSpline,
          typename DensityStateView, typename ForceView>
void launch_coordinate_reverse_edge32(
    const ExecutionSpace& execution_space,
    int persistent_blocks,
    std::size_t num_edges,
    int active_type_count,
    double spline_h,
    double spline_min,
    double cutoff,
    NodeTypesView node_types,
    NeighTypesView neigh_types,
    EdgeReceiversView edge_receivers,
    CoordinateView xyz,
    RadiusView radius,
    bool coordinates_are_unit,
    TypeMapView type_to_active,
    CoefficientsView coefficients,
    bool apply_density_scale,
    ScaleSpline density_scale,
    DensityStateView density_state,
    AdjointView output_adjoint,
    HarmonicsView harmonics_values,
    HarmonicsGradientView harmonics_gradients,
    ForceView forces)
{
    launch_coordinate_reverse_edge_owned<
        edge32_team_size, edge32_vector_length>(
            execution_space, persistent_blocks, num_edges, active_type_count,
            spline_h, spline_min, cutoff, node_types, neigh_types, edge_receivers,
            xyz, radius, coordinates_are_unit, type_to_active, coefficients,
            apply_density_scale,
            density_scale, density_state, output_adjoint, harmonics_values,
            harmonics_gradients, forces);
}

} // namespace symmetrix::standard_r0
