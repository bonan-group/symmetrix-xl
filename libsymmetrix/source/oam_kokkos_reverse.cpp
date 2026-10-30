#include "oam_kokkos.hpp"
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "device_backend.hpp"
#include "oam_validation.hpp"

namespace {

template <typename Precision>
KOKKOS_INLINE_FUNCTION Precision normalized_silu_derivative(const Precision value)
{
    constexpr Precision factor = static_cast<Precision>(1.6791767923989418);
    const Precision sigmoid = value >= Precision(0)
        ? Precision(1) / (Precision(1) + Kokkos::exp(-value))
        : Kokkos::exp(value) / (Precision(1) + Kokkos::exp(value));
    return factor * (sigmoid + value * sigmoid * (Precision(1) - sigmoid));
}

template <typename Precision>
void launch_conditioned_self_reverse(
    Kokkos::View<std::int64_t**, Kokkos::LayoutRight> path_data,
    Kokkos::View<Precision*> coefficients, Kokkos::View<Precision*> wigner,
    Kokkos::View<Precision*> mask, Kokkos::View<int*> type_map,
    Kokkos::View<int*> node_types, const int num_nodes, const int input_width,
    const std::size_t coefficients_per_model, const int path_count,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint)
{
    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    Kokkos::parallel_for(
        "NequIPOAMKokkos conditioned self reverse",
        team_policy(num_nodes, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const typename team_policy::member_type& team) {
            const int node = team.league_rank();
            const int model = type_map(node_types(node));
            const std::size_t model_base = static_cast<std::size_t>(model)
                * coefficients_per_model;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, input_width),
                [&](const int input_lane) {
                    Precision value = Precision(0);
                    for (int path = 0; path < path_count; ++path) {
                        const int input_offset = static_cast<int>(path_data(path, 0));
                        const int input_multiplicity = static_cast<int>(path_data(path, 1));
                        const int width = static_cast<int>(path_data(path, 2));
                        const int output_offset = static_cast<int>(path_data(path, 6));
                        const int output_multiplicity = static_cast<int>(path_data(path, 7));
                        const int relative = input_lane - input_offset;
                        const std::int64_t input_extent = path_data(path, 1)
                            * path_data(path, 2);
                        if (relative < 0
                            || static_cast<std::int64_t>(relative) >= input_extent)
                            continue;
                        const int component = relative / input_multiplicity;
                        const int input_channel = relative % input_multiplicity;
                        const std::size_t coefficient_base = model_base
                            + static_cast<std::size_t>(path_data(path, 10));
                        const Precision angular = wigner(
                            static_cast<std::size_t>(path_data(path, 9))
                            + static_cast<std::size_t>(component));
                        for (int output_channel = 0;
                             output_channel < output_multiplicity; ++output_channel) {
                            const std::size_t coefficient_index = coefficient_base
                                + static_cast<std::size_t>(input_channel)
                                    * static_cast<std::size_t>(output_multiplicity)
                                + static_cast<std::size_t>(output_channel);
                            const int output_index = output_offset
                                + component * output_multiplicity + output_channel;
                            const int mask_index = output_offset
                                + output_channel * width + component;
                            value += output_adjoint(node, output_index)
                                * coefficients(coefficient_index) * angular
                                * mask(mask_index);
                        }
                    }
                    input_adjoint(node, input_lane) += value;
                });
        });
}

}  // namespace

template <typename Precision>
void NequIPOAMKokkos<Precision>::apply_conditioned_self_reverse(
    const LayerState& state,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**, Kokkos::LayoutRight> input_adjoint)
{
    if (!state.conditioned_self.eligible || !execution_space_is_cuda())
        return;
    const auto& plan = state.conditioned_self;
    launch_conditioned_self_reverse<Precision>(plan.path_data, plan.coefficients,
        plan.wigner_diagonal, plan.output_mask, type_to_model_, node_types_,
        num_nodes_, state.input_width, plan.coefficients_per_model,
        plan.path_count, output_adjoint, input_adjoint);
    ++conditioned_self_reverse_launch_count_;
}


template <typename Precision>
void NequIPOAMKokkos<Precision>::run_reverse(const char* operation)
{
    const std::size_t edges = edge_count();
    if (runtime_layers_.size() != layer_states_.size())
        throw std::logic_error("NequIPOAMKokkos reverse state is incomplete.");

    using View2 = Kokkos::View<Precision**, Kokkos::LayoutRight>;
    const int final_width = readout_.input_dimension();
    View2 feature_adjoint = arena_matrix("node_adjoint_a", 0,
        static_cast<std::size_t>(num_nodes_),
        static_cast<std::size_t>(final_width));
    View2 readout_adjoint = arena_matrix("reverse_scratch", 0,
        static_cast<std::size_t>(num_nodes_),
        static_cast<std::size_t>(readout_.output_dimension()));
    Kokkos::deep_copy(readout_adjoint, Precision(0));
    const auto readout_output_adjoint = readout_adjoint;
    const auto readout_types = node_types_;
    const auto readout_type_map = type_to_model_;
    const auto readout_scales = per_type_scale_device_;
    Kokkos::parallel_for(
        "NequIPOAMKokkos initialize readout adjoint",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<int>>(0, num_nodes_),
        KOKKOS_LAMBDA(const int node) {
            const int model_type = readout_type_map(readout_types(node));
            readout_output_adjoint(node, 0) = static_cast<Precision>(
                readout_scales(model_type));
        });
    readout_.reverse_lane(readout_adjoint, feature_adjoint);

    // Retain +dE/dr, including ZBL, for independent force/stress reductions.
    // Capacity is reused and only the active directed-edge extent is cleared.
    symmetrix_oam_detail::ensure_view_capacity(active_geometry_adjoint_,
        "oam directed geometry adjoint", 3 * edges);
    if (edges != 0)
        Kokkos::deep_copy(
            Kokkos::subview(active_geometry_adjoint_,
                std::make_pair(std::size_t(0), 3 * edges)),
            Precision(0));

    bool feature_adjoint_uses_a = true;
    for (std::size_t reverse_index = layer_states_.size(); reverse_index != 0;
         --reverse_index) {
        const std::size_t layer_index = reverse_index - 1;
        auto& state = layer_states_[layer_index];
        auto& runtime = runtime_layers_[layer_index];
        const int input_width = state.input_width;
        const int prelinear_width = state.prelinear_width;
        const int output_width = state.tp_output_width;
        const int pre_gate_width = state.pre_gate_width;
        const int edge_width = state.edge_width;
        const int weight_count = state.weight_count;

        // Retaining every layer's pre-linear checkpoint makes XL capacity
        // scale with their sum. Recompute one layer at a time from its retained
        // input instead; the shared packing arena bounds the extra scratch at
        // the largest single linear.
        std::size_t reverse_cursor = 0;
        View2 prelinear = runtime.prelinear.data() != nullptr
            ? runtime.prelinear
            : arena_matrix("reverse_scratch", reverse_cursor,
                  static_cast<std::size_t>(num_nodes_),
                  static_cast<std::size_t>(prelinear_width));
        if (runtime.prelinear.data() == nullptr) {
            reverse_cursor = symmetrix::oam::checked_oam_add(
                reverse_cursor, prelinear.size());
            state.pre_linear.evaluate_lane(runtime.input, prelinear);
        }

        // Reverse the scalar/gated nonlinearity row by row.  Each row is
        // independent, so no atomics or full edge tape are required here.
        const auto gated_plan = state.gated_block_plan;
        const auto gate_channels = state.gated_gate_channels;
        const int gated_blocks = static_cast<int>(gated_plan.extent(0));
        View2 pre_gate_adjoint = arena_matrix("reverse_scratch", reverse_cursor,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(pre_gate_width));
        reverse_cursor = symmetrix::oam::checked_oam_add(
            reverse_cursor, pre_gate_adjoint.size());
        Kokkos::deep_copy(pre_gate_adjoint, Precision(0));
        const auto pre_gate = runtime.pre_gate;
        const auto output_adjoint = feature_adjoint;
        const int scalar_dimension = state.scalar_dimension;
        const int gate_dimension = state.gate_dimension;
        Kokkos::parallel_for(
            "NequIPOAMKokkos reverse scalar gate",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<int>>(0, num_nodes_),
            KOKKOS_LAMBDA(const int node) {
                for (int component = 0; component < scalar_dimension; ++component)
                    pre_gate_adjoint(node, component) = output_adjoint(node, component)
                        * normalized_silu_derivative(pre_gate(node, component));
                int copy_cursor = 0;
                for (int block = 0; block < gated_blocks; ++block) {
                    const int offset = gated_plan(block, 0);
                    const int multiplicity = gated_plan(block, 1);
                    const int width = gated_plan(block, 2);
                    for (int copy = 0; copy < multiplicity; ++copy) {
                        const int gate_index = gate_channels(copy_cursor++);
                        Precision gate = Precision(1);
                        if (gate_index >= 0) {
                            const Precision value = pre_gate(node, gate_index);
                            gate = (value >= Precision(0)
                                ? value / (Precision(1) + Kokkos::exp(-value))
                                : value * Kokkos::exp(value)
                                    / (Precision(1) + Kokkos::exp(value)))
                                * static_cast<Precision>(1.6791767923989418);
                        }
                        Precision gate_adjoint = Precision(0);
                        for (int component = 0; component < width; ++component) {
                            const int lane =
                                offset + component * multiplicity + copy;
                            const int gated_index =
                                scalar_dimension + gate_dimension + lane;
                            const int output_index = scalar_dimension + lane;
                            pre_gate_adjoint(node, gated_index) +=
                                output_adjoint(node, output_index) * gate;
                            if (gate_index >= 0)
                                gate_adjoint += output_adjoint(node, output_index)
                                    * pre_gate(node, gated_index);
                        }
                        if (gate_index >= 0)
                            pre_gate_adjoint(node, gate_index) += gate_adjoint
                                * normalized_silu_derivative(
                                    pre_gate(node, gate_index));
                    }
                }
            });

        // reverse_lane initializes this entire tensor explicitly. Avoid the
        // redundant Kokkos allocation initializer, whose implicit launch can
        // exceed 2^31 elements for the XL l=4 message representation.
        View2 message_adjoint = arena_matrix("reverse_scratch", reverse_cursor,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(output_width));
        reverse_cursor = symmetrix::oam::checked_oam_add(
            reverse_cursor, message_adjoint.size());
        const auto message_adjoint_data=message_adjoint.data();
        Kokkos::parallel_for(
            "NequIPOAMKokkos initialize message adjoint",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0,message_adjoint.size()),
            KOKKOS_LAMBDA(const std::size_t flat) {
                message_adjoint_data[flat]=Precision(0);
            });
        state.post_linear.reverse_lane(pre_gate_adjoint,message_adjoint,false);
        // The forward scaled every receiver message by the layer output mask
        // and the layer normalization.  The generated reverse callbacks consume
        // that product directly, because their generated forward stores
        // ``normalization * mask * tensor_product``.  The generic tensor product
        // instead applies the output mask inside its own reverse, so the
        // portable path must consume the normalization alone or the mask is
        // applied twice.  The selected edge implementation therefore decides
        // the epilogue scale, and the generated spelling below is unchanged.
        // The edge-owned artifact serves the explicit non-direct modes only,
        // which is the same rule the forward and reverse dispatches apply.
        const bool generated_edge_reverse =
            (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_))
            || receiver_owned_execution()
            || (state.generated_reverse_owner != nullptr && oam_host_plugin_
                && static_cast<bool>(*oam_host_plugin_)
                && streamed_edges_mode_ == "debug_v1");
        {
            const auto output_plan = state.tp_output_block_plan;
            const auto tp_output_mask = state.tp_output_mask;
            const int output_blocks = static_cast<int>(output_plan.extent(0));
            const Precision epilogue_normalization = state.normalization_factor;
            const bool mask_in_epilogue = generated_edge_reverse;
            auto adjoint = message_adjoint;
            Kokkos::parallel_for(
                "NequIPOAMKokkos message adjoint epilogue",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                        0,static_cast<std::size_t>(num_nodes_)),
                KOKKOS_LAMBDA(const std::size_t node) {
                    for (int block = 0; block < output_blocks; ++block) {
                        const int offset = output_plan(block, 0);
                        const int multiplicity = output_plan(block, 1);
                        const int width = output_plan(block, 2);
                        for (int channel = 0; channel < multiplicity; ++channel)
                            for (int component = 0; component < width;
                                ++component) {
                                const int node_index =
                                    offset + channel * width + component;
                                const int lane_index =
                                    offset + component * multiplicity + channel;
                                const Precision scale = mask_in_epilogue
                                    ? tp_output_mask(node_index)
                                        * epilogue_normalization
                                    : epilogue_normalization;
                                adjoint(node, lane_index) *= scale;
                            }
                    }
                });
        }
        View2 input_adjoint = arena_matrix(
            feature_adjoint_uses_a ? "node_adjoint_b" : "node_adjoint_a", 0,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(input_width));
        Kokkos::deep_copy(input_adjoint, Precision(0));

        if (state.self_linear) {
            View2 self_adjoint = arena_matrix("reverse_scratch", reverse_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(input_width));
            reverse_cursor = symmetrix::oam::checked_oam_add(
                reverse_cursor, self_adjoint.size());
            state.self_linear->reverse_lane(pre_gate_adjoint, self_adjoint);
            Kokkos::parallel_for(
                "NequIPOAMKokkos add linear self adjoint",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, input_adjoint.size()),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const std::size_t node = flat / static_cast<std::size_t>(input_width);
                    const std::size_t component = flat % static_cast<std::size_t>(input_width);
                    input_adjoint(node, component) += self_adjoint(node, component);
                });
        } else if (state.conditioned_self.eligible && execution_space_is_cuda()
            && streamed_edges_mode_ == "direct") {
            apply_conditioned_self_reverse(state, pre_gate_adjoint, input_adjoint);
        } else if (state.self_tensor_product) {
            View2 node_attributes = arena_matrix("reverse_scratch", reverse_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(node_attribute_width_));
            reverse_cursor = symmetrix::oam::checked_oam_add(
                reverse_cursor, node_attributes.size());
            gather_attribute_rows(node_attributes, state.self_attribute_block_plan);
            View2 ignored_input_2_adjoint = arena_matrix(
                "reverse_scratch", reverse_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(node_attribute_width_));
            reverse_cursor = symmetrix::oam::checked_oam_add(
                reverse_cursor, ignored_input_2_adjoint.size());
            if (state.self_lane_kernels) {
                state.self_tensor_product->reverse_node_bilinear_uvw_lane(
                    runtime.input, node_attributes, pre_gate_adjoint,
                    input_adjoint, ignored_input_2_adjoint);
            } else {
                // Generic (definition-layout) tensor product: permute the
                // operands and the output adjoint to node-major, reverse, and
                // permute the resulting input adjoint back to lane-major.
                View2 node_input = arena_matrix("reverse_scratch", reverse_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(input_width));
                reverse_cursor = symmetrix::oam::checked_oam_add(
                    reverse_cursor, node_input.size());
                View2 node_attribute_major = arena_matrix(
                    "reverse_scratch", reverse_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(node_attribute_width_));
                reverse_cursor = symmetrix::oam::checked_oam_add(
                    reverse_cursor, node_attribute_major.size());
                View2 node_output_adjoint = arena_matrix(
                    "reverse_scratch", reverse_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(pre_gate_width));
                reverse_cursor = symmetrix::oam::checked_oam_add(
                    reverse_cursor, node_output_adjoint.size());
                View2 node_input_adjoint = arena_matrix(
                    "reverse_scratch", reverse_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(input_width));
                reverse_cursor = symmetrix::oam::checked_oam_add(
                    reverse_cursor, node_input_adjoint.size());
                View2 node_input_2_adjoint = arena_matrix(
                    "reverse_scratch", reverse_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(node_attribute_width_));
                reverse_cursor = symmetrix::oam::checked_oam_add(
                    reverse_cursor, node_input_2_adjoint.size());
                permute_row_layout(
                    runtime.input, node_input, state.input_block_plan, false);
                permute_row_layout(node_attributes, node_attribute_major,
                    state.self_attribute_block_plan, false);
                permute_row_layout(pre_gate_adjoint, node_output_adjoint,
                    state.pre_gate_block_plan, false);
                View2 ignored_weights;
                state.self_tensor_product->reverse(
                    node_input, node_attribute_major, {}, node_output_adjoint,
                    node_input_adjoint, node_input_2_adjoint, ignored_weights);
                permute_row_layout(node_input_adjoint, input_adjoint,
                    state.input_block_plan, true);
            }
        }

        View2 prelinear_adjoint = arena_matrix("reverse_scratch", reverse_cursor,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(prelinear_width));
        reverse_cursor = symmetrix::oam::checked_oam_add(
            reverse_cursor, prelinear_adjoint.size());
        Kokkos::deep_copy(prelinear_adjoint, Precision(0));
        const auto& radial = radial_sets_[layer_index].kokkos_set();
        const auto sources = neigh_indices_;
        const auto receivers = edge_receivers_;
        const auto radii = active_distances_;
        const auto coordinates = active_xyz_;
        const auto harmonics = active_harmonics_;
        const auto harmonic_gradients = active_harmonic_gradients_;
        const auto geometry_adjoint = active_geometry_adjoint_;
        const auto harmonic_indices = state.edge_harmonic_indices;
        const int harmonic_count = edge_harmonic_count_;
        const auto& receiver_offsets_host = first_neigh_host_;
        auto& chunk = runtime_layers_[layer_index];
        bool edge_dispatched = false;
        if constexpr (!symmetrix::execution::host_execution_space<
                Kokkos::DefaultExecutionSpace>) {
            if (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)) {
            const auto& descriptor = cuda_r1_plugin_->descriptor();
            if (state.generated_layer >= descriptor.layer_count
                || descriptor.layers[state.generated_layer].layer_index
                    != static_cast<std::int32_t>(state.generated_layer))
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA reverse metadata is "
                    "inconsistent.");
            const auto& callback_layer = descriptor.layers[state.generated_layer];
            const bool grouped_reverse =
                (descriptor.capabilities
                    & SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1) != 0
                && callback_layer.reverse_group_count > 0;
            if (descriptor.scalar_size != sizeof(Precision)
                || callback_layer.source_dimension != prelinear_width
                || callback_layer.edge_dimension != edge_width
                || callback_layer.weight_dimension != weight_count
                || callback_layer.target_dimension != output_width
                || callback_layer.spline_interval_count != radial.interval_count()
                || callback_layer.spline_function_count != radial.function_count()
                || callback_layer.edge_harmonic_count != edge_harmonic_count_) {
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA reverse callback "
                    "dimensions are inconsistent.");
            }
            if (grouped_reverse) {
                if (callback_layer.reverse_group_count <= 0
                    || callback_layer.reverse_group_flags != 0)
                    throw std::runtime_error(
                        "NequIPOAMKokkos grouped CUDA reverse schedule is invalid.");
                if (static_cast<std::size_t>(callback_layer.reverse_group_count)
                        > std::numeric_limits<std::size_t>::max()
                            / (3 * std::max<std::size_t>(edges, 1)))
                    throw std::length_error(
                        "NequIPOAMKokkos grouped CUDA reverse workspace overflows size_t.");
                const std::size_t required =
                    static_cast<std::size_t>(callback_layer.reverse_group_count)
                    * 3 * edges;
                if (runtime.reverse_geometry_partials.extent(0) < required)
                    throw std::logic_error(
                        "OAM arena reverse-partials slot is smaller than planned");
            }
            // Direct CUDA reverse recomputes harmonic values and gradients in
            // the generated source-owner kernel and accumulates geometry
            // adjoints directly into the retained active buffer.
            SymmetrixOAMCudaLayerArgsV2 callback_args{};
            callback_args.struct_size = sizeof(SymmetrixOAMCudaLayerArgsV2);
            callback_args.flags =
                SYMMETRIX_OAM_CUDA_LAYER_DIRECT_GEOMETRY_V1;
            callback_args.layer_index = state.generated_layer;
            callback_args.reserved = 0u;
            callback_args.num_nodes = static_cast<std::int64_t>(num_nodes_);
            callback_args.num_edges = static_cast<std::int64_t>(edges);
            callback_args.edge_harmonic_count = edge_harmonic_count_;
            callback_args.spline_interval_count = radial.interval_count();
            callback_args.spline_function_count = radial.function_count();
            callback_args.spline_x0 = radial.spline_x0();
            callback_args.spline_h = radial.spline_h();
            callback_args.source_indices =
                reinterpret_cast<const std::int32_t*>(sources.data());
            callback_args.receiver_offsets =
                reinterpret_cast<const std::int32_t*>(first_neigh_.data());
            callback_args.receiver_indices =
                reinterpret_cast<const std::int32_t*>(receivers.data());
            callback_args.source_edge_offsets = reinterpret_cast<const std::int32_t*>(
                source_edge_offsets_.data());
            callback_args.source_edge_indices = reinterpret_cast<const std::int32_t*>(
                source_edge_indices_.data());
            callback_args.distances = radii.data();
            callback_args.spline_coefficients = radial.coefficient_data();
            callback_args.edge_harmonics = grouped_reverse
                ? static_cast<const void*>(active_harmonics_.data()) : nullptr;
            callback_args.edge_vectors = coordinates.data();
            callback_args.target_node_output_adjoint = message_adjoint.data();
            callback_args.source_node_values = prelinear.data();
            callback_args.source_node_input_adjoint = prelinear_adjoint.data();
            callback_args.edge_harmonic_adjoint = nullptr;
            callback_args.distance_adjoint = nullptr;
            callback_args.geometry_adjoint = geometry_adjoint.data();
            callback_args.reverse_segment_count = 0;
            callback_args.reverse_segment_sources = nullptr;
            callback_args.reverse_segment_begin = nullptr;
            callback_args.reverse_segment_end = nullptr;
            callback_args.reverse_segment_order = nullptr;
            callback_args.reverse_source_partials = nullptr;
            callback_args.reverse_geometry_partials = grouped_reverse
                ? static_cast<void*>(runtime.reverse_geometry_partials.data()) : nullptr;
            callback_args.reverse_harmonic_gradients = grouped_reverse
                ? static_cast<const void*>(active_harmonic_gradients_.data()) : nullptr;
            callback_args.runtime_post_linear_weights = nullptr;
            callback_args.runtime_post_linear_bias = nullptr;
            callback_args.normalization_factor =
                static_cast<double>(state.normalization_factor);
#if defined(KOKKOS_ENABLE_CUDA)
            Kokkos::Cuda execution_space;
            void* stream = reinterpret_cast<void*>(execution_space.cuda_stream());
            const auto launch_error = cuda_r1_plugin_->launch_reverse(
                state.generated_layer, &callback_args, stream,
                cuda_r1_grid_blocks_);
            if (launch_error != 0)
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA reverse launch failed "
                    "with error " + std::to_string(launch_error) + ".");
            if (grouped_reverse) {
                const auto geometry_partials = runtime.reverse_geometry_partials;
                const int reverse_groups = callback_layer.reverse_group_count;
                Kokkos::parallel_for(
                    "NequIPOAMKokkos reduce grouped geometry adjoints",
                    Kokkos::RangePolicy<Kokkos::Cuda,
                        Kokkos::IndexType<std::size_t>>(
                        execution_space, 0, 3 * edges),
                    KOKKOS_LAMBDA(const std::size_t index) {
                        Precision value = Precision(0);
                        for (int group = 0; group < reverse_groups; ++group)
                            value += geometry_partials(
                                static_cast<std::size_t>(group) * 3 * edges + index);
                        geometry_adjoint(index) += value;
                    });
            }
            execution_space.fence("Complete NequIP OAM CUDA generated reverse");
            ++cuda_r1_launch_count_;
            ++edge_regions_;
            ++edge_generated_regions_;
            ++edge_dispatches_;
            ++edge_generated_dispatches_;
            ++edge_full_message_launches_;
#else
            throw std::runtime_error(
                "NequIPOAMKokkos device execution requires a CUDA-enabled "
                "Kokkos build.");
#endif
            edge_dispatched = true;
            }
        }
        if constexpr (symmetrix::execution::host_execution_space<
                Kokkos::DefaultExecutionSpace>) {
        if (!edge_dispatched && receiver_owned_execution()) {
            const auto owner = state.source_reverse_owner;
            if (owner == nullptr
                || state.generated_layer != layer_index
                || !oam_host_plugin_v2_ || !*oam_host_plugin_v2_) {
                throw std::runtime_error(
                    "NequIPOAMKokkos source-owned OAM reverse callback is unavailable.");
            }
            const auto& source_owned = oam_host_plugin_v2_->descriptor();
            if (state.generated_layer >= source_owned.layer_count
                || source_owned.layers[state.generated_layer].layer
                    != state.generated_layer)
                throw std::runtime_error(
                    "NequIPOAMKokkos source-owned OAM reverse metadata is inconsistent.");
            const auto& callback_layer = source_owned.layers[state.generated_layer];
            if (source_owned.scalar_size != sizeof(Precision)
                || callback_layer.source_dimension != prelinear_width
                || callback_layer.edge_dimension != edge_width
                || callback_layer.weight_dimension != weight_count
                || callback_layer.target_dimension != output_width) {
                throw std::runtime_error(
                    "NequIPOAMKokkos source-owned OAM reverse callback dimensions "
                    "are inconsistent.");
            }
            // Every directed edge is visited once, source-node adjoints are
            // owner-local, and edge-local geometry partials are overwritten
            // exactly once.
            const std::size_t harmonic_adjoint_size
                = static_cast<std::size_t>(edge_width)
                * static_cast<std::size_t>(edges);
            if (receiver_owned_harmonic_adjoint_.extent(0)
                    < harmonic_adjoint_size
                || receiver_owned_distance_adjoint_.extent(0)
                    < static_cast<std::size_t>(edges))
                throw std::logic_error(
                    "OAM arena source reverse-partial slots are smaller than planned");
            ++edge_regions_;
            ++edge_generated_regions_;
            ++edge_dispatches_;
            ++edge_generated_dispatches_;
            ++edge_full_message_launches_;
            const SymmetrixJitOAMHostSourceReverseArgsV3 callback_args{
                sizeof(SymmetrixJitOAMHostSourceReverseArgsV3),
                state.generated_layer,
                0u,
                static_cast<std::uint32_t>(sizeof(Precision)),
                static_cast<std::int64_t>(num_nodes_),
                static_cast<std::int64_t>(edges),
                static_cast<std::int64_t>(num_nodes_),
                edge_harmonic_count_,
                radial.interval_count(),
                radial.function_count(),
                radial.spline_x0(),
                radial.spline_h(),
                reinterpret_cast<const std::int32_t*>(sources.data()),
                reinterpret_cast<const std::int32_t*>(receivers.data()),
                reinterpret_cast<const std::int32_t*>(source_edge_offsets_.data()),
                reinterpret_cast<const std::int32_t*>(source_edge_indices_.data()),
                radii.data(),
                static_cast<const void*>(radial.coefficient_data()),
                static_cast<const void*>(harmonics.data()),
                static_cast<const void*>(message_adjoint.data()),
                static_cast<const void*>(prelinear.data()),
                static_cast<void*>(prelinear_adjoint.data()),
                static_cast<void*>(receiver_owned_harmonic_adjoint_.data()),
                static_cast<void*>(receiver_owned_distance_adjoint_.data()),
                static_cast<double>(state.normalization_factor),
                nullptr};
            const int owner_width = state.pre_linear.output_dimension();
            const auto owner_work_range =
                symmetrix::execution::checked_oam_owner_work_range(
                    owner_width, state.generated_channel_tile, num_nodes_,
                    state.generated_channel_tiled);
            Kokkos::parallel_for(
                "NequIPOAMKokkos generated OAM source reverse",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<int>>(0, owner_work_range.work_items),
                [=](const int work_item) {
#if defined(__x86_64__) || defined(__i386__)
                    const unsigned int csr = _mm_getcsr();
                    _mm_setcsr(csr | 0x8040u);
                    owner(&callback_args, work_item);
                    _mm_setcsr(csr);
#else
                    owner(&callback_args, work_item);
#endif
                });
            const auto edge_harmonic_adjoint = receiver_owned_harmonic_adjoint_;
            const auto distance_adjoint = receiver_owned_distance_adjoint_;
            Kokkos::parallel_for(
                "NequIPOAMKokkos receiver-owned geometry adjoint",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, edges),
                KOKKOS_LAMBDA(const std::size_t edge) {
                    for (int component = 0; component < 3; ++component) {
                        const Precision radial_force = static_cast<Precision>(
                                distance_adjoint(edge))
                            * static_cast<Precision>(coordinates(3 * edge + component))
                            / static_cast<Precision>(radii(edge));
                        Precision harmonic_force = Precision(0);
                        for (int harmonic = 0; harmonic < edge_width; ++harmonic)
                            harmonic_force += edge_harmonic_adjoint(
                                    static_cast<std::size_t>(edge) * edge_width + harmonic)
                                * harmonic_gradients[
                                    (3 * edge + static_cast<std::size_t>(component))
                                        * harmonic_count
                                    + harmonic_indices(harmonic)];
                        geometry_adjoint(3 * edge + component) +=
                            radial_force + harmonic_force;
                    }
                });
            edge_dispatched = true;
        }
        }
        if (!edge_dispatched) {
            // The portable (T1) compiler-free edge product and the edge-owned
            // ABI v1 artifact share this chunked receiver schedule; only the
            // per-edge kernel differs.  The edge-owned artifact serves the
            // explicit non-direct modes only, matching the forward dispatch.
            bool generated_reverse = false;
            if constexpr (symmetrix::execution::host_execution_space<
                    Kokkos::DefaultExecutionSpace>)
                generated_reverse = state.generated_reverse_owner != nullptr
                    && oam_host_plugin_ && static_cast<bool>(*oam_host_plugin_)
                    && streamed_edges_mode_ == "debug_v1";
            if (generated_reverse) {
                if (state.generated_layer != layer_index) {
                    throw std::runtime_error(
                        "NequIPOAMKokkos generated OAM reverse callback is "
                        "unavailable.");
                }
                const auto& descriptor = oam_host_plugin_->descriptor();
                if (state.generated_layer >= descriptor.layer_count
                    || descriptor.layers[state.generated_layer].layer
                        != state.generated_layer
                    || descriptor.reverse_owners[state.generated_layer]
                        != state.generated_reverse_owner) {
                    throw std::runtime_error(
                        "NequIPOAMKokkos generated OAM reverse callback "
                        "metadata is inconsistent.");
                }
                const auto& callback_layer =
                    descriptor.layers[state.generated_layer];
                if (descriptor.scalar_size != sizeof(Precision)
                    || callback_layer.source_dimension != prelinear_width
                    || callback_layer.edge_dimension != edge_width
                    || callback_layer.weight_dimension != weight_count
                    || callback_layer.output_dimension != output_width) {
                    throw std::runtime_error(
                        "NequIPOAMKokkos generated OAM reverse callback "
                        "dimensions are inconsistent.");
                }
            } else if (state.edge_tensor_product == nullptr) {
                throw std::runtime_error(
                    "NequIPOAMKokkos has no portable or generated OAM reverse "
                    "edge implementation for layer "
                    + std::to_string(layer_index) + ".");
            }
            std::size_t max_chunk_edges = 0;
            for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
                const int receiver_end = std::min(
                    num_nodes_, receiver_begin + static_cast<int>(receiver_chunk_limit_));
                max_chunk_edges = std::max(
                    max_chunk_edges,
                    static_cast<std::size_t>(
                        receiver_offsets_host[static_cast<std::size_t>(receiver_end)]
                        - receiver_offsets_host[static_cast<std::size_t>(receiver_begin)]));
                receiver_begin = receiver_end;
            }
            ensure_chunk_workspace(
                runtime, max_chunk_edges, prelinear_width, edge_width,
                weight_count, output_width);

        for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
            const int receiver_end = std::min(
                num_nodes_, receiver_begin + static_cast<int>(receiver_chunk_limit_));
            const int edge_begin = receiver_offsets_host[static_cast<std::size_t>(receiver_begin)];
            const int edge_end = receiver_offsets_host[static_cast<std::size_t>(receiver_end)];
            const int chunk_edges = edge_end - edge_begin;
            if (chunk_edges != 0) {
                auto edge_features = Kokkos::subview(
                    chunk.chunk_edge_features,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, edge_width));
                auto edge_weights = Kokkos::subview(
                    chunk.chunk_edge_weights,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, weight_count));
                auto edge_output_adjoint = Kokkos::subview(
                    chunk.chunk_output_adjoint,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, output_width));
                Kokkos::parallel_for(
                    "NequIPOAMKokkos gather reverse edge chunk",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0, chunk_edges),
                    KOKKOS_LAMBDA(const int local_edge) {
                        const int edge = edge_begin + local_edge;
                        const int receiver = receivers(edge);
                        for (int component = 0; component < edge_width; ++component)
                            edge_features(local_edge, component) = harmonics(
                                static_cast<std::size_t>(edge) * harmonic_count
                                + harmonic_indices(component));
                        for (int component = 0; component < output_width; ++component)
                            edge_output_adjoint(local_edge, component) =
                                message_adjoint(receiver, component);
                        const auto point = radial.evaluation_point(radii(edge));
                        for (int function = 0; function < weight_count; ++function)
                            edge_weights(local_edge, function) = radial.evaluate_function(
                                0, point, function);
                    });
                auto source_adjoint = Kokkos::subview(
                    chunk.chunk_source_adjoint,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, prelinear_width));
                auto edge_adjoint = Kokkos::subview(
                    chunk.chunk_edge_adjoint,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, edge_width));
                auto weights_adjoint = Kokkos::subview(
                    chunk.chunk_weights_adjoint,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, weight_count));
                const auto run_portable_chunk = [&] {
                    // The generic tensor product addresses node-major rows, so
                    // convert the gathered source and message-adjoint rows,
                    // run the reverse product, then restore lane order.
                    auto source_features = Kokkos::subview(
                        chunk.chunk_source_features,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, prelinear_width));
                    auto source_node = Kokkos::subview(
                        chunk.chunk_source_node,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, prelinear_width));
                    auto output_adjoint_node = Kokkos::subview(
                        chunk.chunk_messages_node,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, output_width));
                    auto source_adjoint_node = Kokkos::subview(
                        chunk.chunk_source_adjoint_node,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, prelinear_width));
                    const auto portable_sources = sources;
                    const auto prelinear_view = prelinear;
                    Kokkos::parallel_for(
                        "NequIPOAMKokkos gather portable reverse source chunk",
                        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                            Kokkos::IndexType<int>>(0, chunk_edges),
                        KOKKOS_LAMBDA(const int local_edge) {
                            const int source = portable_sources(
                                edge_begin + local_edge);
                            for (int component = 0; component < prelinear_width;
                                 ++component)
                                source_features(local_edge, component) =
                                    prelinear_view(source, component);
                        });
                    permute_row_layout(source_features, source_node,
                        state.input_block_plan, false);
                    permute_row_layout(edge_output_adjoint, output_adjoint_node,
                        state.tp_output_block_plan, false);
                    state.edge_tensor_product->reverse(source_node, edge_features,
                        edge_weights, output_adjoint_node, source_adjoint_node,
                        edge_adjoint, weights_adjoint);
                    permute_row_layout(source_adjoint_node, source_adjoint,
                        state.input_block_plan, true);
                    ++edge_regions_;
                    edge_dispatches_ += static_cast<std::size_t>(chunk_edges);
                };
                if constexpr (symmetrix::execution::host_execution_space<
                        Kokkos::DefaultExecutionSpace>) {
                if (generated_reverse) {
                const auto chunk_source_indices = Kokkos::subview(
                    sources, std::make_pair(edge_begin, edge_end));
                const SymmetrixJitOAMHostReverseArgsV1 callback_args{
                    sizeof(SymmetrixJitOAMHostReverseArgsV1),
                    state.generated_layer,
                    sizeof(Precision),
                    0,
                    static_cast<std::int64_t>(edges),
                    static_cast<std::int64_t>(edge_begin),
                    static_cast<std::int64_t>(chunk_edges),
                    reinterpret_cast<const std::int32_t*>(chunk_source_indices.data()),
                    static_cast<const void*>(prelinear.data()),
                    static_cast<const void*>(edge_features.data()),
                    static_cast<const void*>(edge_weights.data()),
                    static_cast<const void*>(edge_output_adjoint.data()),
                    static_cast<void*>(source_adjoint.data()),
                    static_cast<void*>(edge_adjoint.data()),
                    static_cast<void*>(weights_adjoint.data())};
                const auto owner = state.generated_reverse_owner;
                ++edge_regions_;
                ++edge_generated_regions_;
                ++edge_dispatches_;
                ++edge_generated_dispatches_;
                Kokkos::parallel_for(
                    "NequIPOAMKokkos generated OAM tensor-product reverse",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0, chunk_edges),
                    [=](const int local_edge) { owner(&callback_args, local_edge); });
                } else {
                    run_portable_chunk();
                }
                } else {
                    run_portable_chunk();
                }
                Kokkos::parallel_for(
                    "NequIPOAMKokkos scatter reverse edge adjoints",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0, chunk_edges),
                    KOKKOS_LAMBDA(const int local_edge) {
                        const int edge = edge_begin + local_edge;
                        const int source = sources(edge);
                        for (int component = 0; component < prelinear_width; ++component)
                            Kokkos::atomic_add(
                                &prelinear_adjoint(source, component),
                                source_adjoint(local_edge, component));
                        Precision radial_adjoint = Precision(0);
                        const auto point = radial.evaluation_point(radii(edge));
                        for (int function = 0; function < weight_count; ++function) {
                            Precision value = Precision(0);
                            Precision derivative = Precision(0);
                            radial.evaluate_function(
                                0, point, function, value, derivative);
                            radial_adjoint += weights_adjoint(local_edge, function) * derivative;
                        }
                        for (int component = 0; component < 3; ++component) {
                            const Precision radial_force = radial_adjoint
                                * static_cast<Precision>(coordinates(3 * edge + component))
                                / static_cast<Precision>(radii(edge));
                            Precision harmonic_force = Precision(0);
                            for (int harmonic = 0; harmonic < edge_width; ++harmonic)
                                harmonic_force += edge_adjoint(local_edge, harmonic)
                                    * harmonic_gradients[
                                        (3 * static_cast<std::size_t>(edge) + component)
                                            * harmonic_count
                                        + harmonic_indices(harmonic)];
                            geometry_adjoint(3 * static_cast<std::size_t>(edge) + component)
                                += radial_force + harmonic_force;
                        }
                    });
            }
            receiver_begin = receiver_end;
        }
        }

        state.pre_linear.reverse_lane(prelinear_adjoint, input_adjoint, false);
        feature_adjoint = input_adjoint;
        feature_adjoint_uses_a = !feature_adjoint_uses_a;
    }

    // The OAM prior is a radial pair energy. Accumulate its analytic derivative
    // into the resident positive geometry adjoint, shared by forces and stress.
    const auto pair_zsum = zbl_pair_zsum_device_;
    const auto pair_prefactor = zbl_pair_prefactor_device_;
    const std::size_t types = model_atomic_numbers_.size();
    const auto type_map = type_to_model_;
    const auto node_types = node_types_;
    const auto source_types = neigh_types_;
    const auto zbl_receivers = edge_receivers_;
    const auto radii = active_distances_;
    const auto coordinates = active_xyz_;
    const auto geometry_adjoint = active_geometry_adjoint_;
    const auto exponents = zbl_exponents_device_;
    const auto coefficients = zbl_coefficients_device_;
    const double cutoff = zbl_cutoff_;
    const double cutoff_p = zbl_cutoff_p_;
    Kokkos::parallel_for(
        "NequIPOAMKokkos reverse ZBL", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            // Forward energy and the serial reference use double ZBL math
            // with the original double contract coefficients in both model
            // precisions. Round only the final contribution to the geometry
            // adjoint, not the screening constants or analytic derivative.
            const double distance = radii(edge);
            if (!(distance > 0.0) || distance >= cutoff) return;
            const int receiver = zbl_receivers(edge);
            const int model_type_u = type_map(node_types(receiver));
            const int model_type_v = type_map(source_types(edge));
            const std::size_t pair =
                static_cast<std::size_t>(model_type_u) * types + model_type_v;
            constexpr double a0 = 0.46850;
            const double zsum = pair_zsum(pair);
            const double x = zsum * distance / a0;
            const double x_derivative = zsum / a0;
            double screening = 0.0;
            double screening_derivative = 0.0;
            for (std::size_t index = 0; index < exponents.extent(0); ++index) {
                const double term = coefficients(index)
                    * Kokkos::exp(exponents(index) * x);
                screening += term;
                screening_derivative += exponents(index) * x_derivative * term;
            }
            const double normalized = distance / cutoff;
            const double envelope = 1.0
                - ((cutoff_p + 1.0) * (cutoff_p + 2.0) / 2.0)
                    * Kokkos::pow(normalized, cutoff_p)
                + cutoff_p * (cutoff_p + 2.0) * Kokkos::pow(normalized, cutoff_p + 1.0)
                - (cutoff_p * (cutoff_p + 1.0) / 2.0)
                    * Kokkos::pow(normalized, cutoff_p + 2.0);
            const double envelope_derivative = (
                -((cutoff_p + 1.0) * (cutoff_p + 2.0) / 2.0) * cutoff_p
                    * Kokkos::pow(normalized, cutoff_p - 1.0)
                + cutoff_p * (cutoff_p + 2.0) * (cutoff_p + 1.0)
                    * Kokkos::pow(normalized, cutoff_p)
                - (cutoff_p * (cutoff_p + 1.0) / 2.0) * (cutoff_p + 2.0)
                    * Kokkos::pow(normalized, cutoff_p + 1.0))
                / cutoff;
            const double scale = pair_prefactor(pair);
            const double derivative = scale * ((screening_derivative / distance
                - screening / (distance * distance)) * envelope
                + screening / distance * envelope_derivative);
            const Precision force_scale = static_cast<Precision>(derivative / distance);
            for (int component = 0; component < 3; ++component)
                geometry_adjoint(3 * edge + static_cast<std::size_t>(component))
                    += force_scale
                        * static_cast<Precision>(coordinates(3 * edge
                            + static_cast<std::size_t>(component)));
        });

    node_forces.resize(3 * edges);
    if (edges != 0) {
        // Read back only the active extent; the retained adjoint buffer may be
        // larger than the current graph.
        const auto host_geometry = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview(active_geometry_adjoint_,
                std::make_pair(std::size_t(0), 3 * edges)));
        for (std::size_t index = 0; index < 3 * edges; ++index)
            node_forces[index] = -host_geometry(index);
    }
    atom_forces.clear();
    Kokkos::fence();
    if (cuda_r1_plugin_ && *cuda_r1_plugin_) {
        cuda_r1_ready_ = true;
        cuda_r1_reason_ = "generated CUDA energy and force execution completed";
    }
    std::ostringstream diagnostic;
    diagnostic << "NequIP OAM Kokkos execution: backend=" << execution_backend()
        << ", implementation=" << execution_implementation_label() << ", operation=" << operation
        << ", artifact_id=" << execution_artifact_id()
        << ", streamed_edges=" << streamed_edges_mode_
        << ", active_edges=" << edges << ", layers=" << layer_states_.size()
        << ", radial=spline, reverse=analytic, receiver_chunks=" << execution_chunk_count_
        << ", " << execution_kernel_summary();
    last_execution_diagnostic_ = diagnostic.str();
}

template class NequIPOAMKokkos<float>;
template class NequIPOAMKokkos<double>;
