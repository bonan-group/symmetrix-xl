#include "oam_kokkos.hpp"
#if defined(__x86_64__) || defined(__i386__)
#include <xmmintrin.h>
#endif

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "oam_validation.hpp"
#include "spherical_harmonic_device.hpp"
#include "device_backend.hpp"

namespace {

template <typename T>
Kokkos::View<T*> copy_device_vector(const char* label, const std::vector<T>& values)
{
    // The legacy Kokkos::View label constructor, which LAMMPS forces on,
    // accepts a std::string but not a borrowed const char*.
    Kokkos::View<T*> result(std::string(label), values.size());
    auto host = Kokkos::View<T*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        const_cast<T*>(values.data()), values.size());
    Kokkos::deep_copy(result, host);
    return result;
}

template <typename T>
Kokkos::View<T**, Kokkos::LayoutRight> copy_device_matrix(
    const char* label, const std::vector<T>& values, const int rows, const int columns)
{
    Kokkos::View<T**, Kokkos::LayoutRight> result(std::string(label), rows, columns);
    auto host = Kokkos::View<T**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryUnmanaged>(const_cast<T*>(values.data()), rows, columns);
    Kokkos::deep_copy(result, host);
    return result;
}

double zbl_energy(
    const std::vector<int>& model_atomic_numbers, const std::vector<double>& exponents,
    const std::vector<double>& coefficients, const double cutoff, const double cutoff_p,
    const double half_prefactor, const int model_type_u, const int model_type_v,
    const double distance)
{
    if (!(distance > 0.0) || distance >= cutoff) return 0.0;
    const double zi = static_cast<double>(model_atomic_numbers.at(static_cast<std::size_t>(model_type_u)));
    const double zj = static_cast<double>(model_atomic_numbers.at(static_cast<std::size_t>(model_type_v)));
    constexpr double pzbl = 0.23;
    constexpr double a0 = 0.46850;
    const double x = (std::pow(zi, pzbl) + std::pow(zj, pzbl)) * distance / a0;
    double screening = 0.0;
    for (std::size_t index = 0; index < exponents.size(); ++index)
        screening += coefficients[index] * std::exp(exponents[index] * x);
    const double normalized = distance / cutoff;
    const double envelope = 1.0 - ((cutoff_p + 1.0) * (cutoff_p + 2.0) / 2.0)
            * std::pow(normalized, cutoff_p)
        + cutoff_p * (cutoff_p + 2.0) * std::pow(normalized, cutoff_p + 1.0)
        - (cutoff_p * (cutoff_p + 1.0) / 2.0) * std::pow(normalized, cutoff_p + 2.0);
    return half_prefactor * zi * zj / distance * screening * envelope;
}

KOKKOS_INLINE_FUNCTION double normalized_silu(const double value)
{
    constexpr double factor = 1.6791767923989418;
    if (value >= 0.0) {
        const double e = Kokkos::exp(-value);
        return factor * value / (1.0 + e);
    }
    const double e = Kokkos::exp(value);
    return factor * value * e / (1.0 + e);
}

template <typename Precision>
void launch_conditioned_self_forward(
    Kokkos::View<std::int64_t**, Kokkos::LayoutRight> path_data,
    Kokkos::View<Precision*> coefficients, Kokkos::View<Precision*> wigner,
    Kokkos::View<Precision*> mask, Kokkos::View<int*> type_map,
    Kokkos::View<int*> node_types, const int num_nodes,
    const std::size_t coefficients_per_model, const int path_count,
    const int output_width,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
    Kokkos::parallel_for(
        "NequIPOAMKokkos conditioned self forward",
        team_policy(num_nodes, Kokkos::AUTO()),
        KOKKOS_LAMBDA(const typename team_policy::member_type& team) {
            const int node = team.league_rank();
            const int model = type_map(node_types(node));
            const std::size_t model_base = static_cast<std::size_t>(model)
                * coefficients_per_model;
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, output_width),
                [&](const int output_lane) {
                    Precision value = Precision(0);
                    for (int path = 0; path < path_count; ++path) {
                        const int input_offset = static_cast<int>(path_data(path, 0));
                        const int input_multiplicity = static_cast<int>(path_data(path, 1));
                        const int width = static_cast<int>(path_data(path, 2));
                        const int output_offset = static_cast<int>(path_data(path, 6));
                        const int output_multiplicity = static_cast<int>(path_data(path, 7));
                        const int relative = output_lane - output_offset;
                        const std::int64_t output_extent = path_data(path, 7)
                            * path_data(path, 8);
                        if (relative < 0
                            || static_cast<std::int64_t>(relative) >= output_extent)
                            continue;
                        const int component = relative / output_multiplicity;
                        const int output_channel = relative % output_multiplicity;
                        const std::size_t coefficient_base = model_base
                            + static_cast<std::size_t>(path_data(path, 10));
                        const Precision angular = wigner(
                            static_cast<std::size_t>(path_data(path, 9))
                            + static_cast<std::size_t>(component));
                        const Precision output_scale = mask(
                            output_offset + output_channel * width + component);
                        for (int input_channel = 0;
                             input_channel < input_multiplicity; ++input_channel) {
                            const std::size_t coefficient_index = coefficient_base
                                + static_cast<std::size_t>(input_channel)
                                    * static_cast<std::size_t>(output_multiplicity)
                                + static_cast<std::size_t>(output_channel);
                            const int input_lane = input_offset
                                + component * input_multiplicity + input_channel;
                            value += coefficients(coefficient_index)
                                * input(node, input_lane) * angular * output_scale;
                        }
                    }
                    output(node, output_lane) += value;
                });
        });
}

}  // namespace

template <typename Precision>
void NequIPOAMKokkos<Precision>::apply_conditioned_self_forward(
    const LayerState& state,
    Kokkos::View<const Precision**, Kokkos::LayoutRight> input,
    Kokkos::View<Precision**, Kokkos::LayoutRight> output)
{
    if (!state.conditioned_self.eligible || !execution_space_is_cuda())
        return;
    // NVCC extended lambdas require a non-private enclosing launch function.
    const auto& plan = state.conditioned_self;
    launch_conditioned_self_forward<Precision>(plan.path_data, plan.coefficients,
        plan.wigner_diagonal, plan.output_mask, type_to_model_, node_types_,
        num_nodes_, plan.coefficients_per_model, plan.path_count,
        plan.output_width, input, output);
    ++conditioned_self_forward_launch_count_;
}


template <typename Precision>
void NequIPOAMKokkos<Precision>::run_forward(
    const std::span<const double> xyz, const std::span<const double> distances,
    const char* operation, const bool calculate_forces)
{
    completed_generation_ = 0;
    stress_tensor.clear();
    const std::size_t edges = edge_count();
    cuda_r1_ready_ = false;
    if (xyz.size() != 3 * edges || distances.size() != edges)
        throw std::invalid_argument("NequIPOAMKokkos prepared geometry arrays have invalid sizes.");
    if (!nequip_validation::all_finite(xyz) || !nequip_validation::all_finite(distances))
        throw std::invalid_argument("NequIPOAMKokkos xyz and distances must be finite.");
    for (const double distance : distances)
        if (!(distance > 0.0))
            throw std::invalid_argument("NequIPOAMKokkos edge distances must be positive.");
    if (streamed_edges_mode_ == "direct" && !receiver_owned_execution())
        throw std::invalid_argument(
            "NequIP OAM Kokkos direct execution requires an admitted generated "
            "receiver/source-owned ABI v7 host plugin or CUDA plugin; prepare the "
            "generated artifact or select reference mode.");
    if (streamed_edges_mode_ == "debug_v1"
        && !(oam_host_plugin_ && static_cast<bool>(*oam_host_plugin_)))
        throw std::invalid_argument(
            "NequIP OAM Kokkos debug_v1 execution requires an edge-owned ABI v1 "
            "host plugin.");
    // Reference mode owns the portable implementation. An explicit CUDA
    // artifact that failed admission must not be silently replaced.
    if (execution_space_is_cuda() && !cuda_r1_plugin_path_.empty()
        && !(cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)))
        throw std::invalid_argument(
            "NequIP OAM Kokkos rejected the requested CUDA artifact: "
            + cuda_r1_reason_);
    ensure_execution_arena();

    using View2 = Kokkos::View<Precision**, Kokkos::LayoutRight>;
    // The geometry, shuffled-vector, and harmonic buffers are evaluator-owned
    // and capacity-managed.  An evaluation whose graph fits the retained
    // capacity reuses the allocation, and every launch bound below stays the
    // current edge count rather than the capacity.
    symmetrix_oam_detail::ensure_view_capacity(
        active_xyz_, "oam edge vectors", 3 * edges);
    symmetrix_oam_detail::ensure_view_capacity(
        active_distances_, "oam edge distances", edges);
    const auto d_xyz = active_xyz_;
    const auto d_distances = active_distances_;
    auto h_xyz = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        xyz.data(), xyz.size());
    auto h_distances = Kokkos::View<const double*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged>(
        distances.data(), distances.size());
    if (edges != 0) {
        // Copy only the active extent; the retained buffers may be larger.
        auto active_xyz = Kokkos::subview(
            active_xyz_, std::make_pair(std::size_t(0), 3 * edges));
        auto active_distances = Kokkos::subview(
            active_distances_, std::make_pair(std::size_t(0), edges));
        if constexpr (std::is_same_v<Precision, double>) {
            Kokkos::deep_copy(active_xyz, h_xyz);
        } else {
            // Cross-space deep_copy cannot also convert double geometry to float.
            auto converted_xyz = Kokkos::create_mirror_view(active_xyz);
            for (std::size_t index = 0; index < xyz.size(); ++index)
                converted_xyz(index) = static_cast<Precision>(xyz[index]);
            Kokkos::deep_copy(active_xyz, converted_xyz);
        }
        Kokkos::deep_copy(active_distances, h_distances);
    }

    const bool staged_cuda_forward = execution_space_is_cuda()
        && cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)
        && (cuda_r1_plugin_->descriptor().capabilities
            & SYMMETRIX_OAM_CUDA_STAGED_FORWARD_INPUTS_V1) != 0;
    const bool staged_cuda_reverse = execution_space_is_cuda()
        && cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)
        && (cuda_r1_plugin_->descriptor().capabilities
            & SYMMETRIX_OAM_CUDA_GROUPED_REVERSE_PARTIALS_V1) != 0;
    if (staged_cuda_forward && !staged_cuda_reverse) {
        symmetrix_oam_detail::ensure_view_capacity(active_harmonics_,
            "oam staged edge harmonics",
            static_cast<std::size_t>(edge_harmonic_count_) * edges);
        if (edges != 0)
            symmetrix::launch_spherical_harmonic_values_from_directions(
                Kokkos::DefaultExecutionSpace{}, d_xyz, d_distances,
                std::numeric_limits<double>::infinity(), active_harmonics_,
                static_cast<int>(edges), max_edge_l_);
    }

    // Grouped generated reverse and portable dispatches consume evaluator-
    // owned values and gradients. Other generated CUDA reverse kernels keep
    // evaluating them inline inside their source-owner CTA.
    if (staged_cuda_reverse || !execution_space_is_cuda()
        || !(cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_))) {
    symmetrix_oam_detail::ensure_view_capacity(
        active_shuffled_xyz_, "oam shuffled edge vectors", 3 * edges);
    const auto shuffled_xyz = active_shuffled_xyz_;
    Kokkos::parallel_for(
        "NequIPOAMKokkos shuffle edge geometry", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(0, edges),
        KOKKOS_LAMBDA(const std::size_t edge) {
            shuffled_xyz(3 * edge) = d_xyz(3 * edge + 2);
            shuffled_xyz(3 * edge + 1) = d_xyz(3 * edge);
            shuffled_xyz(3 * edge + 2) = d_xyz(3 * edge + 1);
        });
    symmetrix_oam_detail::ensure_view_capacity(active_harmonics_,
        "oam edge harmonics",
        static_cast<std::size_t>(edge_harmonic_count_) * edges);
    symmetrix_oam_detail::ensure_view_capacity(active_harmonic_gradients_,
        "oam edge harmonic gradients",
        3 * static_cast<std::size_t>(edge_harmonic_count_) * edges);
    if (edges != 0) {
        const auto harmonics = active_harmonics_;
        const auto harmonic_gradients = active_harmonic_gradients_;
        const int gradient_harmonic_count = edge_harmonic_count_;
        symmetrix::launch_spherical_harmonics_device(
            Kokkos::DefaultExecutionSpace{}, shuffled_xyz, harmonics, harmonic_gradients,
            static_cast<int>(edges), max_edge_l_);
        const Precision normalization = static_cast<Precision>(2.0 * std::sqrt(std::numbers::pi));
        Kokkos::parallel_for(
            "NequIPOAMKokkos normalize edge harmonics",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, edge_harmonic_count_ * edges),
            KOKKOS_LAMBDA(const std::size_t index) { harmonics(index) *= normalization; });
        // SpheriCart evaluates the shuffled (z, x, y) coordinates.  The
        // serial OAM contract exposes gradients in the original (x, y, z)
        // order and applies the same harmonic normalization to values and
        // gradients.
        Kokkos::parallel_for(
            "NequIPOAMKokkos remap harmonic gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(0, edges * gradient_harmonic_count),
            KOKKOS_LAMBDA(const std::size_t flat) {
                const std::size_t edge = flat / gradient_harmonic_count;
                const std::size_t component = flat % gradient_harmonic_count;
                const std::size_t base = edge * 3 * gradient_harmonic_count;
                const Precision shuffled_x = harmonic_gradients[
                    base + gradient_harmonic_count + component];
                const Precision shuffled_y = harmonic_gradients[
                    base + 2 * gradient_harmonic_count + component];
                const Precision shuffled_z = harmonic_gradients[base + component];
                harmonic_gradients[base + component] = normalization * shuffled_x;
                harmonic_gradients[base + gradient_harmonic_count + component]
                    = normalization * shuffled_y;
                harmonic_gradients[base + 2 * gradient_harmonic_count + component]
                    = normalization * shuffled_z;
            });
    }
    }

    const auto d_sources = neigh_indices_;
    const auto d_receivers = edge_receivers_;
    const auto& receiver_offsets_host = first_neigh_host_;

    const int attribute_width = node_attribute_width_;
    const int harmonic_count = edge_harmonic_count_;
    if (runtime_layers_.size() != layer_states_.size())
        runtime_layers_.resize(layer_states_.size());
    View2 features = runtime_layers_.front().input;
    gather_attribute_rows(features, node_feature_block_plan_);

    std::size_t receiver_chunks = 0;
    for (std::size_t layer_index = 0; layer_index < layer_states_.size(); ++layer_index) {
        auto& state = layer_states_[layer_index];
        auto& runtime = runtime_layers_[layer_index];
        runtime.input = features;
        std::size_t forward_cursor = 0;
        View2 prelinear = runtime.prelinear.data() != nullptr
            ? runtime.prelinear
            : arena_matrix("forward_scratch", forward_cursor,
                  static_cast<std::size_t>(num_nodes_),
                  static_cast<std::size_t>(state.prelinear_width));
        if (runtime.prelinear.data() == nullptr)
            forward_cursor = symmetrix::oam::checked_oam_add(
                forward_cursor, prelinear.size());
        state.pre_linear.evaluate_lane(features, prelinear);
        // Generated receiver owners assign every output lane. Avoid Kokkos'
        // default initialization kernel here: XL message tensors can exceed
        // 2^31 elements even though their byte extent and all explicit launch
        // ranges are 64-bit safe. The portable atomic path initializes this
        // view explicitly below.
        View2 messages = arena_matrix("forward_scratch", forward_cursor,
            static_cast<std::size_t>(num_nodes_),
            static_cast<std::size_t>(state.tp_output_width));
        forward_cursor = symmetrix::oam::checked_oam_add(
            forward_cursor, messages.size());
        const auto& radial = radial_sets_[layer_index].kokkos_set();
        const int edge_width = state.edge_width;
        const int prelinear_width = state.prelinear_width;
        const Precision normalization_factor = state.normalization_factor;
        const int pre_gate_width = state.pre_gate_width;
        const int weight_count = state.weight_count;
        const int output_width = state.tp_output_width;
        bool edge_dispatched = false;
        if constexpr (!symmetrix::execution::host_execution_space<
                Kokkos::DefaultExecutionSpace>) {
            if (cuda_r1_plugin_ && static_cast<bool>(*cuda_r1_plugin_)) {
            const auto& descriptor = cuda_r1_plugin_->descriptor();
            if (state.generated_layer >= descriptor.layer_count
                || descriptor.layers[state.generated_layer].layer_index
                    != static_cast<std::int32_t>(state.generated_layer))
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA forward metadata is "
                    "inconsistent.");
            const auto& callback_layer = descriptor.layers[state.generated_layer];
            if (descriptor.scalar_size != sizeof(Precision)
                || callback_layer.source_dimension != prelinear_width
                || callback_layer.edge_dimension != edge_width
                || callback_layer.weight_dimension != weight_count
                || callback_layer.output_dimension != output_width
                || callback_layer.spline_interval_count != radial.interval_count()
                || callback_layer.spline_function_count != radial.function_count()
                || callback_layer.edge_harmonic_count != edge_harmonic_count_) {
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA forward callback "
                    "dimensions are inconsistent.");
            }
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
                reinterpret_cast<const std::int32_t*>(d_sources.data());
            callback_args.receiver_offsets =
                reinterpret_cast<const std::int32_t*>(first_neigh_.data());
            callback_args.receiver_indices =
                reinterpret_cast<const std::int32_t*>(d_receivers.data());
            callback_args.source_edge_offsets = reinterpret_cast<const std::int32_t*>(
                source_edge_offsets_.data());
            callback_args.source_edge_indices = reinterpret_cast<const std::int32_t*>(
                source_edge_indices_.data());
            callback_args.distances = active_distances_.data();
            callback_args.spline_coefficients = radial.coefficient_data();
            callback_args.edge_harmonics = staged_cuda_forward
                ? static_cast<const void*>(active_harmonics_.data()) : nullptr;
            callback_args.edge_vectors = active_xyz_.data();
            callback_args.source_node_values = prelinear.data();
            callback_args.node_messages = messages.data();
            callback_args.runtime_post_linear_weights = nullptr;
            callback_args.runtime_post_linear_bias = nullptr;
            callback_args.normalization_factor =
                static_cast<double>(normalization_factor);
#if defined(KOKKOS_ENABLE_CUDA)
            Kokkos::Cuda execution_space;
            void* stream = reinterpret_cast<void*>(execution_space.cuda_stream());
            // Forward and reverse have independent persistent grids. sm120
            // FP32 uses the qualified 8/6 CTAs per SM; other targets retain
            // the plugin's reverse default and twice that cap for forward.
            const auto& device = execution_space.cuda_device_prop();
            const std::int64_t compute_units = std::max(1, device.multiProcessorCount);
            const std::int64_t per_unit =
                cuda_r1_plugin_->persistent_blocks_per_compute_unit();
            const std::int64_t reverse_per_unit = qualified_sm120_fp32_ ? 6 : per_unit;
            const std::int64_t forward_per_unit = qualified_sm120_fp32_ ? 8 : 2 * per_unit;
            cuda_r1_grid_blocks_ = static_cast<std::int32_t>(
                std::clamp<std::int64_t>(reverse_per_unit * compute_units,
                    1, std::numeric_limits<std::int32_t>::max()));
            const auto forward_grid_blocks = static_cast<std::int32_t>(
                std::clamp<std::int64_t>(forward_per_unit * compute_units,
                    1, std::numeric_limits<std::int32_t>::max()));
            const auto launch_error = cuda_r1_plugin_->launch_forward(
                state.generated_layer, &callback_args, stream,
                forward_grid_blocks);
            if (launch_error != 0)
                throw std::runtime_error(
                    "NequIPOAMKokkos generated OAM CUDA forward launch failed "
                    "with error " + std::to_string(launch_error) + ".");
            execution_space.fence("Complete NequIP OAM CUDA generated forward");
            ++cuda_r1_launch_count_;
            ++edge_regions_;
            ++edge_generated_regions_;
            ++edge_dispatches_;
            ++edge_generated_dispatches_;
            ++edge_full_message_launches_;
            ++receiver_chunks;
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
            if (state.receiver_forward_owner == nullptr
                || state.generated_layer != layer_index
                || !oam_host_plugin_v2_ || !*oam_host_plugin_v2_) {
                throw std::runtime_error(
                    "NequIPOAMKokkos receiver-owned OAM forward callback is unavailable.");
            }
            const auto& receiver_owned = oam_host_plugin_v2_->descriptor();
            if (state.generated_layer >= receiver_owned.layer_count
                || receiver_owned.layers[state.generated_layer].layer
                    != state.generated_layer)
                throw std::runtime_error(
                    "NequIPOAMKokkos receiver-owned OAM forward metadata is inconsistent.");
            const auto& callback_layer = receiver_owned.layers[state.generated_layer];
            if (receiver_owned.scalar_size != sizeof(Precision)
                || callback_layer.source_dimension != prelinear_width
                || callback_layer.edge_dimension != edge_width
                || callback_layer.weight_dimension != weight_count
                || callback_layer.output_dimension != output_width) {
                throw std::runtime_error(
                    "NequIPOAMKokkos receiver-owned OAM forward callback dimensions "
                    "are inconsistent.");
            }
            const SymmetrixJitOAMHostReceiverForwardArgsV3 callback_args{
                sizeof(SymmetrixJitOAMHostReceiverForwardArgsV3),
                state.generated_layer,
                0u,
                static_cast<std::uint32_t>(sizeof(Precision)),
                static_cast<std::int64_t>(num_nodes_),
                static_cast<std::int64_t>(edges),
                0,
                static_cast<std::int64_t>(edges),
                edge_harmonic_count_,
                radial.interval_count(),
                radial.function_count(),
                radial.spline_x0(),
                radial.spline_h(),
                reinterpret_cast<const std::int32_t*>(d_sources.data()),
                reinterpret_cast<const std::int32_t*>(first_neigh_.data()),
                active_distances_.data(),
                static_cast<const void*>(radial.coefficient_data()),
                static_cast<const void*>(active_harmonics_.data()),
                static_cast<const void*>(prelinear.data()),
                static_cast<void*>(messages.data()),
                static_cast<double>(normalization_factor),
                nullptr,
                nullptr};
            const auto owner = state.receiver_forward_owner;
            const int owner_width = messages.extent_int(1);
            const auto owner_work_range =
                symmetrix::execution::checked_oam_owner_work_range(
                    owner_width, state.generated_channel_tile, num_nodes_,
                    state.generated_channel_tiled);
            const int owner_work_items = owner_work_range.work_items;
            ++edge_regions_;
            ++edge_generated_regions_;
            ++edge_dispatches_;
            ++edge_generated_dispatches_;
            ++edge_full_message_launches_;
            Kokkos::parallel_for(
                "NequIPOAMKokkos generated OAM receiver forward",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<int>>(0, owner_work_items),
                [=](const int work_item) {
#if defined(__x86_64__) || defined(__i386__)
                    // Same subnormal-flush rationale as the source reverse
                    // dispatch; see oam_kokkos_reverse.cpp.
                    const unsigned int csr = _mm_getcsr();
                    _mm_setcsr(csr | 0x8040u);
                    owner(&callback_args, work_item);
                    _mm_setcsr(csr);
#else
                    owner(&callback_args, work_item);
#endif
                });
            ++receiver_chunks;
            edge_dispatched = true;
        }
        }
        if (!edge_dispatched) {
            // The portable (T1) compiler-free edge product and the edge-owned
            // ABI v1 artifact share this chunked receiver schedule; only the
            // per-edge kernel differs.  The edge-owned artifact serves the
            // explicit non-direct modes only. Direct mode is fail-closed before
            // this point when no receiver/source-owned artifact is admitted;
            // only reference mode may reach the portable implementation.
            bool generated_forward = false;
            if constexpr (symmetrix::execution::host_execution_space<
                    Kokkos::DefaultExecutionSpace>)
                generated_forward = state.generated_forward_owner != nullptr
                    && oam_host_plugin_ && static_cast<bool>(*oam_host_plugin_)
                    && streamed_edges_mode_ == "debug_v1";
            if (generated_forward) {
                if (state.generated_layer != layer_index) {
                    throw std::runtime_error(
                        "NequIPOAMKokkos generated OAM forward callback is "
                        "unavailable.");
                }
                const auto& descriptor = oam_host_plugin_->descriptor();
                if (state.generated_layer >= descriptor.layer_count
                    || descriptor.layers[state.generated_layer].layer
                        != state.generated_layer
                    || descriptor.forward_owners[state.generated_layer]
                        != state.generated_forward_owner) {
                    throw std::runtime_error(
                        "NequIPOAMKokkos generated OAM forward callback "
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
                        "NequIPOAMKokkos generated OAM forward callback "
                        "dimensions are inconsistent.");
                }
            } else if (state.edge_tensor_product == nullptr) {
                throw std::runtime_error(
                    "NequIPOAMKokkos has no portable or generated OAM forward "
                    "edge implementation for layer "
                    + std::to_string(layer_index) + ".");
            }
            // Both paths accumulate into the lane-major message rows with
            // atomics, and the v1 chunk owners do too, so the fill is shared.
            Kokkos::deep_copy(messages, Precision(0));
            std::size_t max_chunk_edges = 0;
            for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
                const int receiver_end = std::min(
                    num_nodes_, receiver_begin + static_cast<int>(receiver_chunk_limit_));
                max_chunk_edges = std::max(
                    max_chunk_edges,
                    static_cast<std::size_t>(first_neigh_host_[static_cast<std::size_t>(receiver_end)]
                        - first_neigh_host_[static_cast<std::size_t>(receiver_begin)]));
                receiver_begin = receiver_end;
            }
            ensure_chunk_workspace(
                runtime, max_chunk_edges, prelinear_width, edge_width,
                weight_count, output_width);
            const auto edge_harmonic_indices = state.edge_harmonic_indices;
            for (int receiver_begin = 0; receiver_begin < num_nodes_;) {
            const int receiver_end = std::min(
                num_nodes_, receiver_begin + static_cast<int>(receiver_chunk_limit_));
            const int edge_begin = receiver_offsets_host[static_cast<std::size_t>(receiver_begin)];
            const int edge_end = receiver_offsets_host[static_cast<std::size_t>(receiver_end)];
            const int chunk_edges = edge_end - edge_begin;
            ++receiver_chunks;
            if (chunk_edges != 0) {
                auto source_features = Kokkos::subview(
                    runtime.chunk_source_features,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, prelinear_width));
                auto edge_features = Kokkos::subview(
                    runtime.chunk_edge_features,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, edge_width));
                auto edge_weights = Kokkos::subview(
                    runtime.chunk_edge_weights,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, weight_count));
                auto edge_messages = Kokkos::subview(
                    runtime.chunk_edge_messages,
                    std::make_pair(0, chunk_edges),
                    std::make_pair(0, output_width));
                const auto sources = d_sources;
                const auto chunk_harmonics = active_harmonics_;
                const auto chunk_sources = source_features;
                const auto chunk_edges_view = edge_features;
                const auto chunk_indices = edge_harmonic_indices;
                const auto prelinear_view = prelinear;
                const auto radii = d_distances;
                Kokkos::parallel_for(
                    "NequIPOAMKokkos gather receiver chunk",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0, chunk_edges),
                    KOKKOS_LAMBDA(const int local_edge) {
                        const int edge = edge_begin + local_edge;
                        const int source = sources(edge);
                        for (int component = 0; component < prelinear_width; ++component)
                            chunk_sources(local_edge, component) = prelinear_view(source, component);
                        for (int component = 0; component < edge_width; ++component)
                            chunk_edges_view(local_edge, component) = chunk_harmonics(
                                static_cast<std::size_t>(edge) * harmonic_count
                                + chunk_indices(component));
                    });
                Kokkos::parallel_for(
                    "NequIPOAMKokkos evaluate radial spline weights",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(0,
                        static_cast<std::size_t>(chunk_edges)
                            * static_cast<std::size_t>(weight_count)),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const std::size_t local_edge = flat
                            / static_cast<std::size_t>(weight_count);
                        const std::size_t function = flat
                            % static_cast<std::size_t>(weight_count);
                        Precision value = Precision(0);
                        const auto point = radial.evaluation_point(
                            radii(edge_begin + static_cast<int>(local_edge)));
                        edge_weights(static_cast<int>(local_edge),
                            static_cast<int>(function)) = radial.evaluate_function(
                            0, point, static_cast<int>(function));
                    });
                const auto run_portable_chunk = [&] {
                    // Convert the gathered lane-major source rows to the
                    // node-major order the generic tensor product addresses,
                    // evaluate the whole chunk, and convert the result back to
                    // the lane-major message rows the post-linear consumes.
                    auto source_node = Kokkos::subview(
                        runtime.chunk_source_node,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, prelinear_width));
                    auto messages_node = Kokkos::subview(
                        runtime.chunk_messages_node,
                        std::make_pair(0, chunk_edges),
                        std::make_pair(0, output_width));
                    permute_row_layout(
                        source_features, source_node, state.input_block_plan, false);
                    state.edge_tensor_product->evaluate(
                        source_node, edge_features, edge_weights, messages_node);
                    permute_row_layout(messages_node, edge_messages,
                        state.tp_output_block_plan, true);
                    ++edge_regions_;
                    edge_dispatches_ += static_cast<std::size_t>(chunk_edges);
                };
                if constexpr (symmetrix::execution::host_execution_space<
                        Kokkos::DefaultExecutionSpace>) {
                if (generated_forward) {
                const auto chunk_source_indices = Kokkos::subview(
                    sources, std::make_pair(edge_begin, edge_end));
                const SymmetrixJitOAMHostForwardArgsV1 callback_args{
                    sizeof(SymmetrixJitOAMHostForwardArgsV1),
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
                    static_cast<void*>(edge_messages.data())};
                const auto owner = state.generated_forward_owner;
                ++edge_regions_;
                ++edge_generated_regions_;
                edge_dispatches_ += static_cast<std::size_t>(chunk_edges);
                edge_generated_dispatches_ += static_cast<std::size_t>(chunk_edges);
                Kokkos::parallel_for(
                    "NequIPOAMKokkos generated OAM tensor-product forward",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<int>>(0, chunk_edges),
                    [=](const int local_edge) { owner(&callback_args, local_edge); });
                } else {
                    run_portable_chunk();
                }
                } else {
                    run_portable_chunk();
                }
                const auto receivers = d_receivers;
                auto node_messages = messages;
                Kokkos::parallel_for(
                    "NequIPOAMKokkos scatter receiver messages",
                    Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                        Kokkos::IndexType<std::size_t>>(0,
                        static_cast<std::size_t>(chunk_edges)
                            * static_cast<std::size_t>(output_width)),
                    KOKKOS_LAMBDA(const std::size_t flat) {
                        const std::size_t local_edge = flat
                            / static_cast<std::size_t>(output_width);
                        const std::size_t component = flat
                            % static_cast<std::size_t>(output_width);
                        const int receiver = receivers(
                            edge_begin + static_cast<int>(local_edge));
                        Kokkos::atomic_add(
                            &node_messages(receiver, static_cast<int>(component)),
                            normalization_factor * edge_messages(
                                static_cast<int>(local_edge),
                                static_cast<int>(component)));
                    });
            }
                receiver_begin = receiver_end;
            }
        }

        View2 postlinear = runtime.pre_gate;
        state.post_linear.evaluate_lane(messages, postlinear);
        if (state.self_linear) {
            View2 self = arena_matrix("forward_scratch", forward_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(state.pre_gate_width));
            forward_cursor = symmetrix::oam::checked_oam_add(
                forward_cursor, self.size());
            state.self_linear->evaluate_lane(features, self);
            Kokkos::parallel_for(
                "NequIPOAMKokkos add linear self connection",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, postlinear.size()),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int node = static_cast<int>(flat / pre_gate_width);
                    const int component = static_cast<int>(flat % pre_gate_width);
                    postlinear(node, component) += self(node, component);
                });
        } else if (state.conditioned_self.eligible && execution_space_is_cuda()
            && streamed_edges_mode_ == "direct") {
            apply_conditioned_self_forward(state, features, postlinear);
        } else if (state.self_tensor_product) {
            View2 self = arena_matrix("forward_scratch", forward_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(state.pre_gate_width));
            forward_cursor = symmetrix::oam::checked_oam_add(
                forward_cursor, self.size());
            View2 node_attributes = arena_matrix("forward_scratch", forward_cursor,
                static_cast<std::size_t>(num_nodes_),
                static_cast<std::size_t>(node_attribute_width_));
            forward_cursor = symmetrix::oam::checked_oam_add(
                forward_cursor, node_attributes.size());
            gather_attribute_rows(node_attributes, state.self_attribute_block_plan);
            if (state.self_lane_kernels) {
                state.self_tensor_product->evaluate_node_bilinear_uvw_lane(
                    features, node_attributes, self);
            } else {
                // Generic (definition-layout) tensor product: permute both
                // operands to node-major, evaluate, and permute the result.
                View2 node_features = arena_matrix("forward_scratch", forward_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(state.input_width));
                forward_cursor = symmetrix::oam::checked_oam_add(
                    forward_cursor, node_features.size());
                View2 node_attribute_major = arena_matrix(
                    "forward_scratch", forward_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(node_attribute_width_));
                forward_cursor = symmetrix::oam::checked_oam_add(
                    forward_cursor, node_attribute_major.size());
                View2 node_self = arena_matrix("forward_scratch", forward_cursor,
                    static_cast<std::size_t>(num_nodes_),
                    static_cast<std::size_t>(state.pre_gate_width));
                forward_cursor = symmetrix::oam::checked_oam_add(
                    forward_cursor, node_self.size());
                permute_row_layout(
                    features, node_features, state.input_block_plan, false);
                permute_row_layout(node_attributes, node_attribute_major,
                    state.self_attribute_block_plan, false);
                Kokkos::View<const Precision**, Kokkos::LayoutRight> fixed_weights;
                state.self_tensor_product->evaluate(
                    node_features, node_attribute_major, fixed_weights, node_self);
                permute_row_layout(
                    node_self, self, state.pre_gate_block_plan, true);
            }
            Kokkos::parallel_for(
                "NequIPOAMKokkos add tensor self connection",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(0, postlinear.size()),
                KOKKOS_LAMBDA(const std::size_t flat) {
                    const int node = static_cast<int>(flat / pre_gate_width);
                    const int component = static_cast<int>(flat % pre_gate_width);
                    postlinear(node, component) += self(node, component);
                });
        }
        runtime.pre_gate = postlinear;

        View2 next = runtime.output;
        const auto gated_plan = state.gated_block_plan;
        const auto gate_channels = state.gated_gate_channels;
        const int gated_blocks = static_cast<int>(gated_plan.extent(0));
        const int scalar_dimension = state.scalar_dimension;
        const int gate_dimension = state.gate_dimension;
        Kokkos::parallel_for(
            "NequIPOAMKokkos apply scalar gate", Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<int>>(0, num_nodes_),
            KOKKOS_LAMBDA(const int node) {
                for (int component = 0; component < scalar_dimension; ++component)
                    next(node, component) = static_cast<Precision>(normalized_silu(
                        static_cast<double>(postlinear(node, component))));
                int copy_cursor = 0;
                for (int block = 0; block < gated_blocks; ++block) {
                    const int offset = gated_plan(block, 0);
                    const int multiplicity = gated_plan(block, 1);
                    const int width = gated_plan(block, 2);
                    for (int copy = 0; copy < multiplicity; ++copy) {
                        const int gate_index = gate_channels(copy_cursor++);
                        const Precision gate = gate_index < 0 ? Precision(1)
                            : static_cast<Precision>(normalized_silu(
                                static_cast<double>(postlinear(node, gate_index))));
                        for (int component = 0; component < width; ++component) {
                            const int lane =
                                offset + component * multiplicity + copy;
                            next(node, scalar_dimension + lane) = gate * postlinear(
                                node, scalar_dimension + gate_dimension + lane);
                        }
                    }
                }
            });
        runtime.output = next;
        features = std::move(next);
    }

    View2 readout_values = arena_matrix("forward_scratch", 0,
        static_cast<std::size_t>(num_nodes_),
        static_cast<std::size_t>(readout_.output_dimension()));
    readout_.evaluate_lane(features, readout_values);
    node_energies.resize(static_cast<std::size_t>(num_nodes_));
    if (execution_space_is_cuda() && streamed_edges_mode_ == "direct") {
        if (active_zbl_energies_.extent(0) < edges
            || active_node_energies_.extent(0)
                < static_cast<std::size_t>(num_nodes_))
            throw std::logic_error(
                "OAM arena final-reduction slot is smaller than planned");
        const auto edge_energies = active_zbl_energies_;
        const auto energies = active_node_energies_;
        const auto offsets = first_neigh_;
        const auto node_types = node_types_;
        const auto source_types = neigh_types_;
        const auto type_map = type_to_model_;
        const auto shifts = per_type_shift_device_;
        const auto scales = per_type_scale_device_;
        const auto pair_zsum = zbl_pair_zsum_device_;
        const auto pair_prefactor = zbl_pair_prefactor_device_;
        const auto exponents = zbl_exponents_device_;
        const auto coefficients = zbl_coefficients_device_;
        const std::size_t types = model_atomic_numbers_.size();
        const double cutoff = zbl_cutoff_;
        const double p = zbl_cutoff_p_;
        using team_policy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        Kokkos::parallel_for(
            "NequIPOAMKokkos receiver ZBL energy epilogue",
            team_policy(num_nodes_, Kokkos::AUTO()),
            KOKKOS_LAMBDA(const typename team_policy::member_type& team) {
                const int node = team.league_rank();
                const int model_type = type_map(node_types(node));
                const int begin = offsets(node);
                const int end = offsets(node + 1);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team, begin, end),
                    [&](const int edge) {
                        const double distance = d_distances(edge);
                        double value = 0.0;
                        if (distance > 0.0 && distance < cutoff) {
                            const int source_type = type_map(source_types(edge));
                            const std::size_t pair =
                                static_cast<std::size_t>(model_type) * types + source_type;
                            // Keep multiplication/division order and the envelope
                            // of the OAM prior, not MACE's different ZBL function.
                            const double x = pair_zsum(pair) * distance / 0.46850;
                            double screening = 0.0;
                            for (std::size_t index = 0; index < exponents.extent(0); ++index)
                                screening += coefficients(index)
                                    * Kokkos::exp(exponents(index) * x);
                            const double normalized = distance / cutoff;
                            const double envelope = 1.0
                                - ((p + 1.0) * (p + 2.0) / 2.0)
                                    * Kokkos::pow(normalized, p)
                                + p * (p + 2.0) * Kokkos::pow(normalized, p + 1.0)
                                - (p * (p + 1.0) / 2.0)
                                    * Kokkos::pow(normalized, p + 2.0);
                            value = pair_prefactor(pair) / distance * screening * envelope;
                        }
                        edge_energies(edge) = value;
                    });
                team.team_barrier();
                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    double energy = static_cast<double>(readout_values(node, 0))
                        * scales(model_type) + shifts(model_type);
                    // Preserve receiver order while accumulating physical
                    // energy contributions in FP64.
                    for (int edge = begin; edge < end; ++edge)
                        energy += edge_energies(edge);
                    energies(node) = energy;
                });
            });
        if (num_nodes_ != 0) {
            auto host_energies = Kokkos::View<double*, Kokkos::HostSpace,
                Kokkos::MemoryUnmanaged>(node_energies.data(), node_energies.size());
            Kokkos::deep_copy(host_energies, Kokkos::subview(active_node_energies_,
                std::make_pair(std::size_t(0), static_cast<std::size_t>(num_nodes_))));
        }
    } else {
        // Preserve the established CPU and edge-owned debug execution path.
        auto h_readout = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), readout_values);
        for (int node = 0; node < num_nodes_; ++node) {
            const int model_type = type_to_model_host_.at(
                static_cast<std::size_t>(node_types_host_[node]));
            node_energies[static_cast<std::size_t>(node)] =
                static_cast<double>(h_readout(node, 0))
                    * per_type_scale_.at(static_cast<std::size_t>(model_type))
                + per_type_shift_.at(static_cast<std::size_t>(model_type));
        }
        std::size_t edge = 0;
        for (int receiver = 0; receiver < num_nodes_; ++receiver)
            for (int local = 0; local < num_neigh_host_[static_cast<std::size_t>(receiver)];
                 ++local, ++edge) {
                const int receiver_type = type_to_model_host_.at(
                    static_cast<std::size_t>(node_types_host_[receiver]));
                const int source_type = type_to_model_host_.at(
                    static_cast<std::size_t>(neigh_types_host_[edge]));
                node_energies[static_cast<std::size_t>(receiver)] += zbl_energy(
                    model_atomic_numbers_, zbl_exponents_, zbl_coefficients_,
                    zbl_cutoff_, zbl_cutoff_p_, zbl_half_prefactor_,
                    receiver_type, source_type, distances[edge]);
            }
    }

    execution_chunk_count_ = receiver_chunks;
    std::ostringstream diagnostic;
    diagnostic << "NequIP OAM Kokkos execution: backend=" << execution_backend()
        << ", implementation=" << execution_implementation_label() << ", operation=" << operation
        << ", artifact_id=" << execution_artifact_id()
        << ", streamed_edges=" << streamed_edges_mode_
        << ", active_edges=" << edges << ", layers=" << layer_states_.size()
        << ", radial=spline, receiver_chunks=" << receiver_chunks
        << ", " << execution_kernel_summary();
    last_execution_diagnostic_ = diagnostic.str();
    if (calculate_forces)
        run_reverse(operation);
    else {
        node_forces.clear();
        atom_forces.clear();
        if (cuda_r1_plugin_ && *cuda_r1_plugin_) {
            cuda_r1_ready_ = true;
            cuda_r1_reason_ = "generated CUDA forward execution completed";
            update_diagnostic(operation);
        }
    }
    completed_generation_ = calculate_forces && graph_prepared_ ? geometry_generation_ : 0;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::compute_node_energies_forces(
    const int num_nodes, const std::span<const int> node_types,
    const std::span<const int> num_neigh, const std::span<const int> neigh_indices,
    const std::span<const int> neigh_types, const std::span<const double> xyz,
    const std::span<const double> distances)
{
    prepare_graph(num_nodes, node_types, num_neigh, neigh_indices, neigh_types);
    run_forward(xyz, distances, "direct");
    graph_prepared_ = false;
    geometry_prepared_ = false;
    completed_generation_ = 0;
}

template <typename Precision>
void NequIPOAMKokkos<Precision>::compute_prepared(
    const std::uint64_t generation, const std::span<const double> xyz,
    const std::span<const double> distances)
{
    validate_geometry_generation(generation);
    run_forward(xyz, distances, "compute_prepared");
    completed_generation_ = generation;
}

template class NequIPOAMKokkos<float>;
template class NequIPOAMKokkos<double>;
