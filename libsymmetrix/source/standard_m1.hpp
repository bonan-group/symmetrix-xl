#pragma once

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "standard_m0.hpp"

namespace symmetrix::standard_m1 {

inline constexpr int input_components = standard_m0::input_components;
inline constexpr int total_terms = standard_m0::term_offsets[1];
inline constexpr int default_host_channel_tile = 16;
inline constexpr int device_overlap_channel_tile = 32;

template <typename ExecutionSpace>
inline constexpr bool supports_device_direct_reverse()
{
#if defined(KOKKOS_ENABLE_HIP)
    if constexpr (std::is_same_v<ExecutionSpace,Kokkos::HIP>)
        return true;
#endif
#if defined(KOKKOS_ENABLE_CUDA)
    if constexpr (std::is_same_v<ExecutionSpace,Kokkos::Cuda>)
        return true;
#endif
    return false;
}

template <typename Scalar, typename ExecutionSpace>
inline constexpr bool supports_backend()
{
    if constexpr (!std::is_same_v<Scalar,float>)
        return false;
    if constexpr (std::is_same_v<
            typename ExecutionSpace::memory_space, Kokkos::HostSpace>)
        return true;
    if constexpr (supports_device_direct_reverse<ExecutionSpace>())
        return true;
    return false;
}

template <typename LeftView, typename RightView>
inline bool views_overlap(const LeftView& left, const RightView& right)
{
    if (left.data() == nullptr || right.data() == nullptr
        || left.span() == 0 || right.span() == 0)
        return false;
    const auto left_begin = reinterpret_cast<std::uintptr_t>(left.data());
    const auto right_begin = reinterpret_cast<std::uintptr_t>(right.data());
    const std::size_t left_bytes = left.span()
        *sizeof(typename LeftView::non_const_value_type);
    const std::size_t right_bytes = right.span()
        *sizeof(typename RightView::non_const_value_type);
    return left_begin <= right_begin
        ? right_begin-left_begin < left_bytes
        : left_begin-right_begin < right_bytes;
}

inline bool matches_structure(
    int runtime_input_components,
    const std::vector<std::vector<int>>& runtime_terms,
    std::vector<int>* canonical_rows = nullptr)
{
    if (runtime_input_components != input_components
        || runtime_terms.size() != static_cast<std::size_t>(total_terms))
        return false;
    auto rows = std::vector<int>(runtime_terms.size());
    std::iota(rows.begin(), rows.end(), 0);
    std::sort(rows.begin(), rows.end(), [&] (const int left, const int right) {
        const auto& lhs = runtime_terms[left];
        const auto& rhs = runtime_terms[right];
        if (lhs.size() != rhs.size())
            return lhs.size() < rhs.size();
        return lhs < rhs;
    });
    for (int term=0; term<total_terms; ++term) {
        const auto& components = runtime_terms[rows[term]];
        if (!standard_m0::matches_term(
                0, term,
                std::span<const int>(components.data(), components.size())))
            return false;
    }
    if (canonical_rows != nullptr)
        *canonical_rows = std::move(rows);
    return true;
}

inline int host_channel_tile()
{
    static const int tile = [] {
        const char* value = std::getenv("SYMMETRIX_STANDARD_M1_HOST_TILE");
        if (value == nullptr || std::string_view(value).empty())
            return default_host_channel_tile;
        if (std::string_view(value) == "runtime")
            return 0;
        if (std::string_view(value) == "1")
            return 1;
        if (std::string_view(value) == "4")
            return 4;
        if (std::string_view(value) == "8")
            return 8;
        if (std::string_view(value) == "16")
            return 16;
        return default_host_channel_tile;
    }();
    return tile;
}

template <int Tile, typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView, typename OutputView>
void launch_forward_host_tiled(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputView output)
{
    using Scalar = typename InputView::non_const_value_type;
    Kokkos::parallel_for(
        "StandardM1::forward_host_tiled",
        Kokkos::RangePolicy<ExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space, 0, static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t node) {
            const int type = node_types(node);
            for (int channel_begin=0; channel_begin<channel_count;
                 channel_begin+=Tile) {
                const int active_channels = Kokkos::min(
                    Tile, channel_count-channel_begin);
                Scalar input_values[input_components][Tile];
                Scalar output_values[standard_m0::output_components][Tile] = {};
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int lane=0; lane<active_channels; ++lane)
                    for (int component=0; component<input_components; ++component)
                        input_values[component][lane] =
                            input(node,component,channel_begin+lane);
                standard_m0::accumulate_forward_terms<Tile>(
                    type, channel_begin, active_channels,
                    input_values, weights, output_values,
                    std::make_index_sequence<total_terms>{});
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int lane=0; lane<active_channels; ++lane)
                    output(node,channel_begin+lane) = output_values[0][lane];
            }
        });
}

template <int Tile, typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
void launch_reverse_host_tiled(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputAdjointView output_adjoint,
    const InputAdjointView input_adjoint,
    const InputScaleAdjointView input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    using Scalar = typename InputView::non_const_value_type;
    Kokkos::parallel_for(
        "StandardM1::reverse_host_tiled",
        Kokkos::RangePolicy<ExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            execution_space, 0, static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t node) {
            const int type = node_types(node);
            double scale_adjoint = 0.0;
            for (int channel_begin=0; channel_begin<channel_count;
                 channel_begin+=Tile) {
                const int active_channels = Kokkos::min(
                    Tile, channel_count-channel_begin);
                Scalar input_values[input_components][Tile];
                Scalar output_adjoints[standard_m0::output_components][Tile] = {};
                Scalar input_adjoints[input_components][Tile] = {};
#if defined(_OPENMP)
#pragma omp simd
#endif
                for (int lane=0; lane<active_channels; ++lane) {
                    for (int component=0; component<input_components; ++component)
                        input_values[component][lane] =
                            input(node,component,channel_begin+lane);
                    output_adjoints[0][lane] =
                        output_adjoint(node,channel_begin+lane);
                }
                standard_m0::accumulate_reverse_terms<Tile>(
                    type, channel_begin, active_channels,
                    input_values, output_adjoints, weights, input_adjoints,
                    std::make_index_sequence<total_terms>{});
                for (int component=0; component<input_components; ++component) {
#if defined(_OPENMP)
#pragma omp simd reduction(+:scale_adjoint)
#endif
                    for (int lane=0; lane<active_channels; ++lane) {
                        input_adjoint(node,component,channel_begin+lane) =
                            input_adjoints[component][lane];
                        if (capture_input_scale_adjoint)
                            scale_adjoint += static_cast<double>(
                                input_values[component][lane])
                                *input_adjoints[component][lane];
                    }
                }
            }
            if (capture_input_scale_adjoint)
                input_scale_adjoint(node) = scale_adjoint;
        });
}

template <int Component, std::size_t Term,
          typename InputView, typename WeightsView, typename Scalar>
KOKKOS_INLINE_FUNCTION void accumulate_reverse_component_term(
    const std::size_t node,
    const int type,
    const int channel,
    const InputView& input,
    const WeightsView& weights,
    Scalar& input_adjoint)
{
    constexpr int degree = standard_m0::term_degrees[Term];
    constexpr int component_0 = standard_m0::term_components[Term][0];
    constexpr int component_1 = standard_m0::term_components[Term][1];
    constexpr int component_2 = standard_m0::term_components[Term][2];
    if constexpr (Component == component_0
            || Component == component_1
            || Component == component_2) {
        Scalar derivative = 0;
        if constexpr (degree == 1) {
            derivative = 1;
        } else if constexpr (degree == 2) {
            if constexpr (Component == component_0)
                derivative += input(node,component_1,channel);
            if constexpr (Component == component_1)
                derivative += input(node,component_0,channel);
        } else {
            if constexpr (Component == component_0)
                derivative += input(node,component_1,channel)
                    *input(node,component_2,channel);
            if constexpr (Component == component_1)
                derivative += input(node,component_0,channel)
                    *input(node,component_2,channel);
            if constexpr (Component == component_2)
                derivative += input(node,component_0,channel)
                    *input(node,component_1,channel);
        }
        input_adjoint += weights(type,Term,channel)*derivative;
    }
}

template <int Component, typename InputView, typename WeightsView,
          std::size_t... Terms>
KOKKOS_INLINE_FUNCTION typename InputView::non_const_value_type
reverse_component(
    const std::size_t node,
    const int type,
    const int channel,
    const InputView& input,
    const WeightsView& weights,
    std::index_sequence<Terms...>)
{
    using Scalar = typename InputView::non_const_value_type;
    Scalar input_adjoint = 0;
    (accumulate_reverse_component_term<Component,Terms>(
        node, type, channel, input, weights, input_adjoint), ...);
    return input_adjoint;
}

template <typename InputView, typename WeightsView>
KOKKOS_INLINE_FUNCTION typename InputView::non_const_value_type
reverse_component_runtime(
    const int component,
    const std::size_t node,
    const int type,
    const int channel,
    const InputView& input,
    const WeightsView& weights)
{
    using Terms = std::make_index_sequence<total_terms>;
    switch (component) {
    case 0: return reverse_component<0>(node,type,channel,input,weights,Terms{});
    case 1: return reverse_component<1>(node,type,channel,input,weights,Terms{});
    case 2: return reverse_component<2>(node,type,channel,input,weights,Terms{});
    case 3: return reverse_component<3>(node,type,channel,input,weights,Terms{});
    case 4: return reverse_component<4>(node,type,channel,input,weights,Terms{});
    case 5: return reverse_component<5>(node,type,channel,input,weights,Terms{});
    case 6: return reverse_component<6>(node,type,channel,input,weights,Terms{});
    case 7: return reverse_component<7>(node,type,channel,input,weights,Terms{});
    case 8: return reverse_component<8>(node,type,channel,input,weights,Terms{});
    case 9: return reverse_component<9>(node,type,channel,input,weights,Terms{});
    case 10: return reverse_component<10>(node,type,channel,input,weights,Terms{});
    case 11: return reverse_component<11>(node,type,channel,input,weights,Terms{});
    case 12: return reverse_component<12>(node,type,channel,input,weights,Terms{});
    case 13: return reverse_component<13>(node,type,channel,input,weights,Terms{});
    case 14: return reverse_component<14>(node,type,channel,input,weights,Terms{});
    case 15: return reverse_component<15>(node,type,channel,input,weights,Terms{});
    default: return 0;
    }
}

#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
template <typename ValuesView>
struct DeviceChannelTileInput {
    using non_const_value_type = typename ValuesView::non_const_value_type;

    ValuesView values;
    int channel_begin;

    KOKKOS_INLINE_FUNCTION non_const_value_type operator()(
        const std::size_t, const int component, const int channel) const
    {
        return values(component, channel-channel_begin);
    }
};

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
void launch_reverse_device_direct(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputAdjointView output_adjoint,
    const InputAdjointView input_adjoint,
    const InputScaleAdjointView input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    using Scalar = typename InputView::non_const_value_type;
    const std::size_t work = static_cast<std::size_t>(num_nodes)
        *input_components*static_cast<std::size_t>(channel_count);
    Kokkos::parallel_for(
        "StandardM1::reverse_device_direct",
        Kokkos::RangePolicy<ExecutionSpace,Kokkos::IndexType<std::size_t>>(
            execution_space, 0, work),
        KOKKOS_LAMBDA (const std::size_t index) {
            const int channel = static_cast<int>(index%channel_count);
            const std::size_t node_component = index/channel_count;
            const int component = static_cast<int>(
                node_component%input_components);
            const std::size_t node = node_component/input_components;
            const int type = node_types(node);
            Scalar adjoint = 0;
            switch (component) {
            case 0:
                adjoint = reverse_component<0>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 1:
                adjoint = reverse_component<1>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 2:
                adjoint = reverse_component<2>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 3:
                adjoint = reverse_component<3>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 4:
                adjoint = reverse_component<4>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 5:
                adjoint = reverse_component<5>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 6:
                adjoint = reverse_component<6>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 7:
                adjoint = reverse_component<7>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 8:
                adjoint = reverse_component<8>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 9:
                adjoint = reverse_component<9>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 10:
                adjoint = reverse_component<10>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 11:
                adjoint = reverse_component<11>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 12:
                adjoint = reverse_component<12>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 13:
                adjoint = reverse_component<13>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 14:
                adjoint = reverse_component<14>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            case 15:
                adjoint = reverse_component<15>(node, type, channel, input,
                    weights, std::make_index_sequence<total_terms>{}); break;
            }
            input_adjoint(node,component,channel) =
                adjoint*output_adjoint(node,channel);
        });
    if (!capture_input_scale_adjoint)
        return;

    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamMember = typename TeamPolicy::member_type;
    Kokkos::parallel_for(
        "StandardM1::reverse_device_scale_adjoint",
        TeamPolicy(execution_space, num_nodes, 8, 32),
        KOKKOS_LAMBDA (const TeamMember& member) {
            const std::size_t node =
                static_cast<std::size_t>(member.league_rank());
            double scale_adjoint = 0;
            Kokkos::parallel_reduce(
                Kokkos::TeamVectorRange(
                    member, input_components*channel_count),
                [=] (const int item, double& local_adjoint) {
                    const int component = item/channel_count;
                    const int channel = item%channel_count;
                    local_adjoint += static_cast<double>(
                        input(node,component,channel))
                        *input_adjoint(node,component,channel);
                }, scale_adjoint);
            Kokkos::single(Kokkos::PerTeam(member), [=] () {
                input_scale_adjoint(node) = scale_adjoint;
            });
        });
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
void launch_reverse_device_overlap_safe(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputAdjointView output_adjoint,
    const InputAdjointView input_adjoint,
    const InputScaleAdjointView input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    using Scalar = typename InputView::non_const_value_type;
    using TeamPolicy = Kokkos::TeamPolicy<ExecutionSpace>;
    using TeamMember = typename TeamPolicy::member_type;
    using ScratchView = Kokkos::View<
        Scalar**,Kokkos::LayoutRight,
        typename TeamMember::scratch_memory_space,Kokkos::MemoryUnmanaged>;
    const int channel_tile = device_overlap_channel_tile;
    auto policy = TeamPolicy(execution_space, num_nodes, 4, 32);
    policy.set_scratch_size(0, Kokkos::PerTeam(
        ScratchView::shmem_size(input_components, channel_tile)));
    Kokkos::parallel_for(
        "StandardM1::reverse_device_overlap_safe", policy,
        KOKKOS_LAMBDA (const TeamMember& member) {
            const std::size_t node =
                static_cast<std::size_t>(member.league_rank());
            const int type = node_types(node);
            ScratchView input_values(
                member.team_scratch(0), input_components,
                channel_tile);
            if (capture_input_scale_adjoint)
                Kokkos::single(Kokkos::PerTeam(member), [=] () {
                    input_scale_adjoint(node) = 0.0;
                });
            member.team_barrier();
            for (int channel_begin=0; channel_begin<channel_count;
                 channel_begin+=channel_tile) {
                const int active_channels = Kokkos::min(
                    channel_tile,channel_count-channel_begin);
                Kokkos::parallel_for(
                    Kokkos::TeamVectorRange(
                    member, input_components*active_channels),
                    [=] (const int item) {
                        const int component = item/active_channels;
                        const int lane = item%active_channels;
                        input_values(component,lane) =
                            input(node,component,channel_begin+lane);
                    });
                member.team_barrier();
                const DeviceChannelTileInput<ScratchView> staged_input{
                    input_values, channel_begin};
                double tile_scale_adjoint = 0.0;
                Kokkos::parallel_reduce(
                    Kokkos::TeamVectorRange(
                        member, input_components*active_channels),
                    [=] (const int item, double& local_scale_adjoint) {
                        const int component = item/active_channels;
                        const int lane = item%active_channels;
                        const int channel = channel_begin+lane;
                        const Scalar adjoint = reverse_component_runtime(
                            component, 0, type, channel, staged_input, weights)
                            *output_adjoint(node,channel);
                        if (capture_input_scale_adjoint)
                            local_scale_adjoint += static_cast<double>(
                                input_values(component,lane))*adjoint;
                        input_adjoint(node,component,channel) = adjoint;
                    }, tile_scale_adjoint);
                if (capture_input_scale_adjoint)
                    Kokkos::single(Kokkos::PerTeam(member), [=] () {
                        input_scale_adjoint(node) += tile_scale_adjoint;
                    });
                member.team_barrier();
            }
        });
}
#endif

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView, typename OutputView>
bool launch_forward(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputView output)
{
    if constexpr (!std::is_same_v<
            typename ExecutionSpace::memory_space, Kokkos::HostSpace>) {
        return false;
    } else {
        switch (host_channel_tile()) {
        case 1:
            launch_forward_host_tiled<1>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output);
            return true;
        case 4:
            launch_forward_host_tiled<4>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output);
            return true;
        case 8:
            launch_forward_host_tiled<8>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output);
            return true;
        case 16:
            launch_forward_host_tiled<16>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output);
            return true;
        default:
            return false;
        }
    }
}

template <typename ExecutionSpace, typename NodeTypesView,
          typename InputView, typename WeightsView,
          typename OutputAdjointView, typename InputAdjointView,
          typename InputScaleAdjointView>
bool launch_reverse(
    const ExecutionSpace& execution_space,
    const int num_nodes,
    const int channel_count,
    const NodeTypesView node_types,
    const InputView input,
    const WeightsView weights,
    const OutputAdjointView output_adjoint,
    const InputAdjointView input_adjoint,
    const InputScaleAdjointView input_scale_adjoint,
    const bool capture_input_scale_adjoint)
{
    if constexpr (std::is_same_v<
            typename ExecutionSpace::memory_space, Kokkos::HostSpace>) {
        switch (host_channel_tile()) {
        case 1:
            launch_reverse_host_tiled<1>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
            return true;
        case 4:
            launch_reverse_host_tiled<4>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
            return true;
        case 8:
            launch_reverse_host_tiled<8>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
            return true;
        case 16:
            launch_reverse_host_tiled<16>(
                execution_space, num_nodes, channel_count,
                node_types, input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
            return true;
        default:
            return false;
        }
    }
#if defined(KOKKOS_ENABLE_HIP) || defined(KOKKOS_ENABLE_CUDA)
    else if constexpr (supports_device_direct_reverse<ExecutionSpace>()
            && std::is_same_v<
                typename InputView::non_const_value_type,float>) {
        if (views_overlap(input, input_adjoint)) {
            if (input.data() != input_adjoint.data()
                || input.span() != input_adjoint.span()
                || input.extent(0) != input_adjoint.extent(0)
                || input.extent(1) != input_adjoint.extent(1)
                || input.extent(2) != input_adjoint.extent(2))
                return false;
            launch_reverse_device_overlap_safe(
                execution_space, num_nodes, channel_count, node_types,
                input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
        } else {
            launch_reverse_device_direct(
                execution_space, num_nodes, channel_count, node_types,
                input, weights, output_adjoint, input_adjoint,
                input_scale_adjoint, capture_input_scale_adjoint);
        }
        return true;
    }
#endif
    else {
        return false;
    }
}

} // namespace symmetrix::standard_m1
