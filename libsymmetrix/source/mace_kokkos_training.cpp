#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "KokkosBlas.hpp"
#include "KokkosBlas_tpl_spec.hpp"
#if defined(KOKKOS_ENABLE_HIP) \
    && defined(KOKKOSKERNELS_ENABLE_TPL_ROCBLAS)
#include <rocblas/rocblas.h>
#endif

#include "mace_kokkos.hpp"
#include "standard_m0.hpp"

namespace {

bool finite_double_bits(const double value)
{
    constexpr std::uint64_t exponent_mask = 0x7ff0000000000000ULL;
    return (std::bit_cast<std::uint64_t>(value)&exponent_mask)
        != exponent_mask;
}

std::size_t checked_element_count(const std::vector<std::size_t>& shape)
{
    std::size_t elements = 1;
    for (const std::size_t extent : shape) {
        if (extent != 0
                && elements
                    > std::numeric_limits<std::size_t>::max()/extent)
            throw std::length_error(
                "Direct parameter-gradient result size overflow.");
        elements *= extent;
    }
    return elements;
}

void require_finite(const std::vector<double>& values)
{
    for (const double value : values)
        if (!std::isfinite(value))
            throw std::invalid_argument(
                "Direct parameters must contain only finite values.");
}

}  // namespace

template <typename Precision>
void MACEKokkos<Precision>::validate_direct_parameter_profile() const
{
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Direct parameter gradients require streamed_edges='direct'.");
    if (single_layer_readout)
        throw std::invalid_argument(
            "Direct parameter gradients do not support single-layer MACE.");
    constexpr bool host_execution = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if (has_field_coupling)
        throw std::invalid_argument(
            "Direct parameter gradients currently require ordinary MACE.");
    if (first_interaction_residual)
        throw std::invalid_argument(
            "Direct parameter gradients do not support residual-first MACE.");
    if (!uses_compact_radial || !compact_radial_model || !supports_factorized())
        throw std::invalid_argument(
            "Direct parameter gradients require an admitted compact model.");
    if (host_execution ? !jit_host_plugin_ready() : !jit_device_plugin_ready())
        throw std::invalid_argument(
            host_execution
                ? "Direct parameter gradients require the loaded host JIT plugin."
                : "Direct parameter gradients require the loaded device JIT plugin.");
    if (edge_geometry_policy == EdgeGeometryPolicy::unit_f32_radius_f64)
        throw std::invalid_argument(
            "Compact edge geometry does not support direct parameter gradients.");
    if (low_memory_enabled() || use_mh0_adjoint_reuse() || readout_recompute
            || phi1_policy != Phi1Policy::retained
            || harmonic_storage_policy != HarmonicStoragePolicy::retained
            || m1_polynomial_policy == M1PolynomialPolicy::recompute)
        throw std::invalid_argument(
            "Direct parameter gradients require fully retained node state.");
}

template <typename Precision>
void MACEKokkos<Precision>::build_direct_parameter_groups(
    const std::size_t max_bytes)
{
    const auto runtime_parameters = get_direct_parameters();
    auto groups = std::vector<DirectTrainingParameterGroup>();
    std::size_t result_bytes = 0;
    std::size_t result_elements = 0;
    const auto add_group = [&] (
        std::string name,
        std::string layout,
        std::vector<std::size_t> shape) {
        const std::size_t elements = checked_element_count(shape);
        if (elements
                > std::numeric_limits<std::size_t>::max()/sizeof(Precision))
            throw std::length_error(
                "Direct parameter-gradient result size overflow.");
        const std::size_t bytes = elements*sizeof(Precision);
        if (bytes > max_bytes-result_bytes)
            throw std::length_error(
                "Direct parameter gradients require more than the configured limit of "
                +std::to_string(max_bytes)+" host bytes.");
        const std::size_t offset = result_elements;
        result_bytes += bytes;
        result_elements += elements;
        groups.push_back({
            std::move(name), std::move(layout), std::move(shape),
            offset, elements});
    };

    const std::size_t species = atomic_numbers_host.size();
    const std::size_t channels = static_cast<std::size_t>(num_channels);
    for (int lm=0; lm<num_LM; ++lm) {
        add_group(
            "M0_weights.LM"+std::to_string(lm),
            "type,channel,term",
            {species, channels,
             static_cast<std::size_t>(
                 M0_term_coefficient_nodes[lm].size())});
    }
    for (int l=0; l<=L_max; ++l)
        add_group(
            "H1_weights.l"+std::to_string(l),
            "input_channel,output_channel",
            {channels, channels});
    for (int l=0; l<=l_max; ++l)
        add_group(
            "A1_weights.l"+std::to_string(l),
            "path_input_channel,output_channel",
            {static_cast<std::size_t>(A1_weights(l).extent(0)), channels});
    add_group(
        "M1_weights", "type,channel,term",
        {species, channels,
         static_cast<std::size_t>(M1_term_coefficient_nodes.size())});
    add_group(
        "H2_weights_for_H1", "type,input_channel,output_channel",
        {species, channels, channels});
    add_group(
        "H2_weights_for_M1", "input_channel,output_channel",
        {channels, channels});
    add_group("readout_1_weights", "channel", {channels});

    const auto readout_weights = readout_2.get_weights();
    if (readout_weights.size() != 2)
        throw std::runtime_error(
            "Direct parameter gradients require the two-layer nonlinear readout.");
    const std::size_t hidden = readout_weights[1].size();
    add_group(
        "readout_2.weights.0", "hidden,input_channel", {hidden, channels});
    add_group("readout_2.weights.1", "hidden", {hidden});

    direct_parameter_gradients_max_bytes = max_bytes;
    direct_parameter_gradients_result_bytes = result_bytes;
    direct_parameter_gradient_groups = std::move(groups);
    direct_training_parameters = Kokkos::View<double*>(
        "direct_training_parameters", result_elements);
    direct_training_gradients = Kokkos::View<Precision*>(
        "direct_training_gradients", result_elements);
    Kokkos::deep_copy(factorized_execution_space, direct_training_gradients, 0.0);
    direct_optimizer_group_by_parameter = {};
    direct_optimizer_learning_rates = {};
    direct_optimizer_weight_decays = {};
    direct_optimizer_momentum_buffer = {};
    direct_optimizer_first_moment = {};
    direct_optimizer_second_moment = {};
    direct_optimizer_max_second_moment = {};
    direct_optimizer_name.clear();
    direct_optimizer_steps = 0;

    direct_M0_term_nodes.clear();
    direct_M0_poly_sources.clear();
    direct_standard_M0_sources.clear();
    direct_M0_term_nodes.reserve(num_LM);
    direct_M0_poly_sources.reserve(num_LM);
    direct_standard_M0_sources.reserve(num_LM);
    for (int lm=0; lm<num_LM; ++lm) {
        Kokkos::View<int*> nodes(
            "direct_M0_term_nodes", M0_term_coefficient_nodes[lm].size());
        auto host_nodes = Kokkos::create_mirror_view(nodes);
        for (std::size_t index=0; index<M0_term_coefficient_nodes[lm].size(); ++index)
            host_nodes(index) = M0_term_coefficient_nodes[lm][index];
        Kokkos::deep_copy(nodes, host_nodes);
        direct_M0_term_nodes.push_back(std::move(nodes));

        Kokkos::View<int*> poly_sources(
            "direct_M0_poly_sources", M0_poly_coeff(lm).extent(1));
        auto host_poly_sources = Kokkos::create_mirror_view(poly_sources);
        Kokkos::deep_copy(host_poly_sources, -1);
        for (std::size_t term=0;
                term<M0_term_coefficient_nodes[lm].size(); ++term) {
            const int node = M0_term_coefficient_nodes[lm][term];
            if (node >= 0)
                host_poly_sources(node) = static_cast<int>(term);
        }
        Kokkos::deep_copy(poly_sources, host_poly_sources);
        direct_M0_poly_sources.push_back(std::move(poly_sources));

        const auto& sources = standard_m0_module_ready
            ? standard_m0_canonical_rows[lm] : std::vector<int>();
        Kokkos::View<int*> module_sources(
            "direct_standard_M0_sources", sources.size());
        auto host_sources = Kokkos::create_mirror_view(module_sources);
        for (std::size_t index=0; index<sources.size(); ++index)
            host_sources(index) = sources[index];
        Kokkos::deep_copy(module_sources, host_sources);
        direct_standard_M0_sources.push_back(std::move(module_sources));
    }
    direct_A1_lme_begins.assign(static_cast<std::size_t>(l_max+1), 0);
    const auto host_Phi1_l = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), Phi1_l);
    for (int l=0; l<=l_max; ++l) {
        int lme_begin = 0;
        for (std::size_t path=0; path<host_Phi1_l.extent(0); ++path)
            if (host_Phi1_l(path) < l)
                lme_begin += 2*host_Phi1_l(path)+1;
        direct_A1_lme_begins[static_cast<std::size_t>(l)] = lme_begin;
    }
    direct_M1_term_nodes = Kokkos::View<int*>(
        "direct_M1_term_nodes", M1_term_coefficient_nodes.size());
    auto host_M1_nodes = Kokkos::create_mirror_view(direct_M1_term_nodes);
    for (std::size_t index=0; index<M1_term_coefficient_nodes.size(); ++index)
        host_M1_nodes(index) = M1_term_coefficient_nodes[index];
    Kokkos::deep_copy(direct_M1_term_nodes, host_M1_nodes);
    direct_M1_poly_sources = Kokkos::View<int*>(
        "direct_M1_poly_sources", M1_poly_coeff.extent(1));
    auto host_M1_poly_sources = Kokkos::create_mirror_view(
        direct_M1_poly_sources);
    Kokkos::deep_copy(host_M1_poly_sources, -1);
    for (std::size_t term=0; term<M1_term_coefficient_nodes.size(); ++term) {
        const int node = M1_term_coefficient_nodes[term];
        if (node >= 0)
            host_M1_poly_sources(node) = static_cast<int>(term);
    }
    Kokkos::deep_copy(direct_M1_poly_sources, host_M1_poly_sources);
    const std::size_t M1_source_count = standard_m1_module_ready
        ? standard_m1_canonical_rows.size()
        : static_cast<std::size_t>(M1_weights.extent(1));
    direct_M1_weight_sources = Kokkos::View<int*>(
        "direct_M1_weight_sources", M1_source_count);
    auto host_M1_sources = Kokkos::create_mirror_view(
        direct_M1_weight_sources);
    for (std::size_t index=0; index<M1_source_count; ++index)
        host_M1_sources(index) = standard_m1_module_ready
            ? standard_m1_canonical_rows[index] : static_cast<int>(index);
    Kokkos::deep_copy(direct_M1_weight_sources, host_M1_sources);

    initialize_direct_training_parameters(runtime_parameters);
}

template <typename Precision>
void MACEKokkos<Precision>::set_direct_parameter_gradients(
    const bool enabled,
    const std::size_t max_bytes)
{
    if (!enabled) {
        const bool profile_changed = direct_parameter_gradients_enabled;
        direct_parameter_gradients_enabled = false;
        direct_parameter_gradients_ready = false;
        direct_parameter_gradients_result_bytes = 0;
        std::vector<DirectTrainingParameterGroup>().swap(
            direct_parameter_gradient_groups);
        direct_training_parameters = {};
        direct_training_gradients = {};
        direct_training_gradient_scratch = {};
        direct_training_gradient_accumulator = {};
        direct_A1_packed_inputs = {};
        direct_A1_packed_adjoints = {};
        direct_H1_packed_inputs = {};
        direct_H1_packed_adjoints = {};
        direct_mlp_packed_inputs = {};
        direct_mlp_packed_hidden_derivatives = {};
        direct_training_gradient_accumulator_ready = false;
        direct_training_gradient_accumulator_objective =
            DirectGradientObjective::total_energy;
        direct_training_gradient_accumulator_displacement = 0.0;
        direct_training_gradient_accumulator_batch_size = 0;
        direct_gradient_objective = DirectGradientObjective::total_energy;
        direct_gradient_finite_difference_displacement = 0.0;
        direct_optimizer_group_by_parameter = {};
        direct_optimizer_learning_rates = {};
        direct_optimizer_weight_decays = {};
        direct_optimizer_momentum_buffer = {};
        direct_optimizer_first_moment = {};
        direct_optimizer_second_moment = {};
        direct_optimizer_max_second_moment = {};
        direct_M0_term_nodes.clear();
        direct_M0_poly_sources.clear();
        direct_standard_M0_sources.clear();
        direct_M1_term_nodes = {};
        direct_M1_poly_sources = {};
        direct_M1_weight_sources = {};
        direct_node_energy_adjoints = {};
        direct_batch_structure_offsets = {};
        direct_batch_reference_energies = {};
        direct_batch_energy_residual_scales = {};
        direct_batch_energy_values = {};
        direct_force_positions = {};
        direct_force_cells = {};
        direct_force_edge_shifts = {};
        direct_force_pbc = {};
        direct_force_edge_structures = {};
        direct_force_references = {};
        direct_force_direction = {};
        direct_force_negative_xyz = {};
        direct_force_negative_r = {};
        direct_force_positive_xyz = {};
        direct_force_positive_r = {};
        direct_training_host_to_device_bytes = 0;
        direct_training_device_to_host_bytes = 0;
        direct_training_fallback_count = 0;
        direct_training_fallback_reason.clear();
        direct_optimizer_name.clear();
        direct_optimizer_steps = 0;
        if (profile_changed) {
            factorized_schedule_dirty = true;
            invalidate_factorized_prepared_graph();
        }
        plan_factorized_reverse_cache();
        return;
    }
    if (max_bytes == 0)
        throw std::invalid_argument(
            "Direct parameter-gradient byte limit must be positive.");
    validate_direct_parameter_profile();
    build_direct_parameter_groups(max_bytes);
    direct_training_gradient_accumulator_ready = false;
    direct_training_gradient_accumulator_objective =
        DirectGradientObjective::total_energy;
    direct_training_gradient_accumulator_displacement = 0.0;
    direct_training_gradient_accumulator_batch_size = 0;
    direct_parameter_gradients_ready = false;
    direct_gradient_objective = DirectGradientObjective::total_energy;
    direct_gradient_finite_difference_displacement = 0.0;
    const bool profile_changed = !direct_parameter_gradients_enabled;
    direct_parameter_gradients_enabled = true;
    direct_parameter_gradients_ready = false;
    if (profile_changed) {
        factorized_schedule_dirty = true;
        invalidate_factorized_prepared_graph();
    }
    plan_factorized_reverse_cache();
}

template <typename Precision>
DirectTrainingParameterGroup&
MACEKokkos<Precision>::direct_parameter_gradient_group(
    const std::string& name)
{
    const auto group = std::find_if(
        direct_parameter_gradient_groups.begin(),
        direct_parameter_gradient_groups.end(),
        [&] (const DirectTrainingParameterGroup& candidate) {
            return candidate.name == name;
        });
    if (group == direct_parameter_gradient_groups.end())
        throw std::runtime_error(
            "Direct parameter-gradient group is missing: "+name+".");
    return *group;
}

template <typename Precision>
const DirectTrainingParameterGroup&
MACEKokkos<Precision>::direct_parameter_gradient_group(
    const std::string& name) const
{
    const auto group = std::find_if(
        direct_parameter_gradient_groups.begin(),
        direct_parameter_gradient_groups.end(),
        [&] (const DirectTrainingParameterGroup& candidate) {
            return candidate.name == name;
        });
    if (group == direct_parameter_gradient_groups.end())
        throw std::runtime_error(
            "Direct parameter-gradient group is missing: "+name+".");
    return *group;
}

template <typename Precision>
void MACEKokkos<Precision>::begin_direct_parameter_gradients()
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    Kokkos::deep_copy(
        factorized_execution_space, direct_training_gradients, 0.0);
    direct_parameter_gradients_ready = false;
    direct_batch_node_seed_active = false;
    direct_batch_energies.clear();
    direct_batch_loss = 0.0;
    direct_batch_size = 0;
    direct_gradient_objective = DirectGradientObjective::total_energy;
    direct_gradient_finite_difference_displacement = 0.0;
}

template <typename Precision>
void MACEKokkos<Precision>::finish_direct_parameter_gradients(
    const bool record_capture)
{
    if (direct_parameter_gradients_enabled
            && direct_parameter_gradient_capture_active) {
        if (record_capture) {
            if (direct_parameter_gradient_capture_count
                    == std::numeric_limits<std::size_t>::max())
                throw std::overflow_error(
                    "Direct parameter-gradient capture count overflowed.");
            direct_parameter_gradient_capture_count += 1;
        }
        direct_parameter_gradients_ready = true;
    }
}

template <typename Precision>
void MACEKokkos<Precision>::invalidate_direct_training_state()
{
    direct_parameter_gradients_ready = false;
    direct_training_gradient_accumulator_ready = false;
    direct_training_gradient_accumulator_objective =
        DirectGradientObjective::total_energy;
    direct_training_gradient_accumulator_displacement = 0.0;
    direct_training_gradient_accumulator_batch_size = 0;
    direct_gradient_objective = DirectGradientObjective::total_energy;
    direct_gradient_finite_difference_displacement = 0.0;
    direct_batch_node_seed_active = false;
    direct_batch_size = 0;
    direct_batch_loss = 0.0;
    direct_batch_energies.clear();
    if (direct_training_gradients.extent(0) != 0)
        Kokkos::deep_copy(
            factorized_execution_space, direct_training_gradients, 0.0);
}

template <typename Precision>
void MACEKokkos<Precision>::scale_direct_parameter_gradients(const double weight)
{
    if (!direct_parameter_gradients_ready)
        throw std::logic_error(
            "Scaling direct parameter gradients requires ready gradients.");
    if (!std::isfinite(weight) || weight < 0.0)
        throw std::invalid_argument(
            "Direct parameter-gradient weight must be finite and non-negative.");
    if (weight > static_cast<double>(std::numeric_limits<Precision>::max()))
        throw std::invalid_argument(
            "Direct parameter-gradient weight is not representable.");
    const Precision working_weight = static_cast<Precision>(weight);
    const auto gradients = direct_training_gradients;
    Kokkos::parallel_for(
        "Scale direct parameter gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index) {
            gradients(index) *= working_weight;
        });
    direct_gradient_objective = DirectGradientObjective::weighted_energy_force;
}

template <typename Precision>
void MACEKokkos<Precision>::stash_direct_parameter_gradients(const double weight)
{
    if (!direct_parameter_gradients_ready)
        throw std::logic_error(
            "Stashing direct parameter gradients requires ready gradients.");
    if (!std::isfinite(weight) || weight < 0.0)
        throw std::invalid_argument(
            "Direct parameter-gradient weight must be finite and non-negative.");
    if (weight > static_cast<double>(std::numeric_limits<Precision>::max()))
        throw std::invalid_argument(
            "Direct parameter-gradient weight is not representable.");
    const Precision working_weight = static_cast<Precision>(weight);
    if (direct_training_gradient_accumulator.extent(0)
            != direct_training_gradients.extent(0)) {
        const std::size_t required = direct_training_gradients.extent(0);
        if (required
                > std::numeric_limits<std::size_t>::max()/sizeof(Precision))
            throw std::length_error("Direct training workspace size overflow.");
        std::size_t projected_workspace_bytes = direct_training_workspace_bytes();
        projected_workspace_bytes -=
            direct_training_gradient_accumulator.extent(0)*sizeof(Precision);
        const std::size_t required_bytes = required*sizeof(Precision);
        if (required_bytes
                > std::numeric_limits<std::size_t>::max()
                    -projected_workspace_bytes)
            throw std::length_error("Direct training workspace size overflow.");
        projected_workspace_bytes += required_bytes;
        if (projected_workspace_bytes > direct_parameter_gradients_max_bytes)
            throw std::length_error(
                "Direct training workspace exceeds the configured parameter-gradient "
                "byte limit.");
        direct_training_gradient_accumulator = Kokkos::View<Precision*>(
            "direct_training_gradient_accumulator", required);
    }
    const auto gradients = direct_training_gradients;
    const auto accumulator = direct_training_gradient_accumulator;
    Kokkos::parallel_for(
        "Stash weighted direct parameter gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index) {
            accumulator(index) = working_weight*gradients(index);
        });
    direct_training_gradient_accumulator_objective = direct_gradient_objective;
    direct_training_gradient_accumulator_displacement =
        direct_gradient_finite_difference_displacement;
    direct_training_gradient_accumulator_batch_size = direct_batch_size;
    direct_training_gradient_accumulator_ready = true;
}

template <typename Precision>
void MACEKokkos<Precision>::combine_stashed_direct_parameter_gradients(
    const double weight)
{
    if (!direct_training_gradient_accumulator_ready
            || !direct_parameter_gradients_ready)
        throw std::logic_error(
            "Combining direct parameter gradients requires stashed and ready "
            "gradients.");
    if (!std::isfinite(weight) || weight < 0.0)
        throw std::invalid_argument(
            "Direct parameter-gradient weight must be finite and non-negative.");
    if (weight > static_cast<double>(std::numeric_limits<Precision>::max()))
        throw std::invalid_argument(
            "Direct parameter-gradient weight is not representable.");
    const Precision working_weight = static_cast<Precision>(weight);
    if (direct_training_gradient_accumulator_batch_size != direct_batch_size)
        throw std::invalid_argument(
            "Combined direct parameter gradients have different batch sizes.");
    const bool energy_force =
        (direct_training_gradient_accumulator_objective
                == DirectGradientObjective::mean_half_squared_force_finite_difference
            && direct_gradient_objective
                == DirectGradientObjective::mean_half_squared_energy)
        || (direct_training_gradient_accumulator_objective
                == DirectGradientObjective::mean_half_squared_energy
            && direct_gradient_objective
                == DirectGradientObjective::mean_half_squared_force_finite_difference);
    if (!energy_force)
        throw std::invalid_argument(
            "Combined direct parameter gradients require one energy and one force "
            "objective.");
    const auto gradients = direct_training_gradients;
    const auto accumulator = direct_training_gradient_accumulator;
    Kokkos::parallel_for(
        "Combine weighted direct parameter gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index) {
            gradients(index) = accumulator(index)
                +working_weight*gradients(index);
        });
    if (direct_gradient_finite_difference_displacement == 0.0)
        direct_gradient_finite_difference_displacement =
            direct_training_gradient_accumulator_displacement;
    direct_gradient_objective = DirectGradientObjective::weighted_energy_force;
    direct_training_gradient_accumulator_ready = false;
}

template <typename Precision>
void MACEKokkos<Precision>::initialize_direct_training_parameters(
    const std::map<std::string,std::vector<double>>& runtime_parameters)
{
    auto host = Kokkos::create_mirror_view(direct_training_parameters);
    for (const auto& group : direct_parameter_gradient_groups) {
        const auto found = runtime_parameters.find(group.name);
        if (found == runtime_parameters.end()
                || found->second.size() != group.elements)
            throw std::runtime_error(
                "Direct training parameter initialization is inconsistent for '"
                +group.name+"'.");
        std::copy(
            found->second.begin(), found->second.end(),
            host.data()+group.offset);
    }
    Kokkos::deep_copy(
        factorized_execution_space, direct_training_parameters, host);
}

template <typename Precision>
void MACEKokkos<Precision>::apply_direct_batch_loss_seed(
    const int num_nodes,
    const std::span<const std::size_t> loss_structure_offsets,
    const std::span<const double> loss_reference_energies,
    const std::span<const double> loss_energy_residual_scales)
{
    if (!direct_parameter_gradients_enabled)
        throw std::logic_error(
            "A direct batch loss seed requires direct parameter gradients.");
    if (num_nodes < 0)
        throw std::invalid_argument(
            "Direct batch loss seed node count cannot be negative.");
    if (loss_reference_energies.empty()
            || loss_structure_offsets.size()
                != loss_reference_energies.size()+1)
        throw std::invalid_argument(
            "Direct batch offsets and reference energies are inconsistent.");
    if (loss_structure_offsets.front() != 0
            || loss_structure_offsets.back()
                != static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "Direct batch offsets do not cover the prepared node range.");
    for (std::size_t index=1; index<loss_structure_offsets.size(); ++index)
        if (loss_structure_offsets[index]
                <= loss_structure_offsets[index-1])
            throw std::invalid_argument(
                "Direct batch structures must be non-empty and ordered.");
    for (const double reference : loss_reference_energies)
        if (!finite_double_bits(reference))
            throw std::invalid_argument(
                "Direct batch reference energies must be finite.");
    if (!loss_energy_residual_scales.empty()
            && loss_energy_residual_scales.size()
                != loss_reference_energies.size())
        throw std::invalid_argument(
            "Direct batch energy residual scales must match the batch size.");
    for (const double scale : loss_energy_residual_scales)
        if (!finite_double_bits(scale) || scale <= 0.0)
            throw std::invalid_argument(
                "Direct batch energy residual scales must be finite and positive.");
    if (H2_adj.data() != nullptr && H2_adj.data() == H2.data())
        throw std::logic_error(
            "Direct batch loss seeds require distinct H2 forward and adjoint "
            "storage.");

    const auto node_energies = this->node_energies;
    if (node_energies.extent(0)
            < static_cast<std::size_t>(num_nodes))
        throw std::logic_error(
            "Direct batch node energies do not cover the prepared graph.");

    const std::size_t batch_size = loss_reference_energies.size();
    std::size_t projected_workspace_bytes = direct_training_workspace_bytes();
    const auto project_reallocation = [&] (
        const auto& view, const std::size_t required) {
        if (view.extent(0) == required)
            return;
        using View = std::decay_t<decltype(view)>;
        constexpr std::size_t element_bytes =
            sizeof(typename View::non_const_value_type);
        if (required > std::numeric_limits<std::size_t>::max()/element_bytes)
            throw std::length_error("Direct training workspace size overflow.");
        const std::size_t current_bytes = view.extent(0)*element_bytes;
        const std::size_t required_bytes = required*element_bytes;
        projected_workspace_bytes -= current_bytes;
        if (required_bytes
                > std::numeric_limits<std::size_t>::max()
                    -projected_workspace_bytes)
            throw std::length_error("Direct training workspace size overflow.");
        projected_workspace_bytes += required_bytes;
    };
    project_reallocation(direct_batch_structure_offsets, batch_size+1);
    project_reallocation(direct_batch_reference_energies, batch_size);
    project_reallocation(direct_batch_energy_residual_scales, batch_size);
    project_reallocation(direct_batch_energy_values, batch_size);
    project_reallocation(
        direct_node_energy_adjoints, static_cast<std::size_t>(num_nodes));
    if (projected_workspace_bytes > direct_parameter_gradients_max_bytes)
        throw std::length_error(
            "Direct training workspace exceeds the configured parameter-gradient "
            "byte limit.");
    if (direct_batch_structure_offsets.extent(0) != batch_size+1)
        Kokkos::realloc(direct_batch_structure_offsets, batch_size+1);
    if (direct_batch_reference_energies.extent(0) != batch_size)
        Kokkos::realloc(direct_batch_reference_energies, batch_size);
    if (direct_batch_energy_residual_scales.extent(0) != batch_size)
        Kokkos::realloc(direct_batch_energy_residual_scales, batch_size);
    if (direct_batch_energy_values.extent(0) != batch_size)
        Kokkos::realloc(direct_batch_energy_values, batch_size);
    if (direct_node_energy_adjoints.extent(0)
            != static_cast<std::size_t>(num_nodes))
        Kokkos::realloc(
            direct_node_energy_adjoints,
            static_cast<std::size_t>(num_nodes));
    // Preserve a previous ready capture until all loss workspace is admitted.
    begin_direct_parameter_gradients();
    auto offsets_host = Kokkos::create_mirror_view(
        direct_batch_structure_offsets);
    auto references_host = Kokkos::create_mirror_view(
        direct_batch_reference_energies);
    auto scales_host = Kokkos::create_mirror_view(
        direct_batch_energy_residual_scales);
    std::copy(
        loss_structure_offsets.begin(), loss_structure_offsets.end(),
        offsets_host.data());
    std::copy(
        loss_reference_energies.begin(), loss_reference_energies.end(),
        references_host.data());
    if (loss_energy_residual_scales.empty())
        std::fill(scales_host.data(), scales_host.data()+batch_size, 1.0);
    else
        std::copy(
            loss_energy_residual_scales.begin(),
            loss_energy_residual_scales.end(),
            scales_host.data());
    Kokkos::deep_copy(
        factorized_execution_space,
        direct_batch_structure_offsets,
        offsets_host);
    Kokkos::deep_copy(
        factorized_execution_space,
        direct_batch_reference_energies,
        references_host);
    Kokkos::deep_copy(
        factorized_execution_space,
        direct_batch_energy_residual_scales,
        scales_host);
    constexpr bool device_execution = !std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if constexpr (device_execution)
        direct_training_host_to_device_bytes +=
            sizeof(std::size_t)*loss_structure_offsets.size()
            +sizeof(double)*loss_reference_energies.size()
            +sizeof(double)*batch_size;

    const auto structure_offsets = direct_batch_structure_offsets;
    const auto reference_energies = direct_batch_reference_energies;
    const auto residual_scales = direct_batch_energy_residual_scales;
    const auto structure_energies = direct_batch_energy_values;
    Kokkos::parallel_for(
        "Reduce direct batch structure energies",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, batch_size),
        KOKKOS_LAMBDA (const std::size_t structure) {
            double energy = 0.0;
            for (std::size_t node=structure_offsets(structure);
                    node<structure_offsets(structure+1); ++node)
                energy += node_energies(node);
            structure_energies(structure) = energy;
        });
    factorized_execution_space.fence("Direct batch structure-energy reduction");
    const auto energies_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), structure_energies);
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(double)*batch_size;
    direct_batch_energies.assign(
        energies_host.data(), energies_host.data()+batch_size);
    for (std::size_t structure=0; structure<batch_size; ++structure) {
        const double energy = direct_batch_energies[structure];
        if (!finite_double_bits(energy))
            throw std::invalid_argument(
                "Direct batch structure energy is non-finite.");
        const double scale = loss_energy_residual_scales.empty()
            ? 1.0 : loss_energy_residual_scales[structure];
        const double scaled_error =
            scale*(energy-loss_reference_energies[structure]);
        const double loss_term = 0.5*scaled_error*scaled_error;
        const double seed = scaled_error*scale
            /static_cast<double>(batch_size);
        if (!finite_double_bits(loss_term) || !finite_double_bits(seed))
            throw std::invalid_argument(
                "Direct batch scaled loss or seed is non-finite.");
        if (std::abs(seed)
                > static_cast<double>(std::numeric_limits<Precision>::max()))
            throw std::invalid_argument(
                std::string("Direct batch loss seed is not representable in ")
                +(std::is_same_v<Precision,float> ? "float32" : "float64")
                +" working precision.");
        direct_batch_loss += loss_term;
        if (!finite_double_bits(direct_batch_loss))
            throw std::invalid_argument(
                "Direct batch accumulated loss is non-finite.");
    }
    direct_batch_loss /= static_cast<double>(batch_size);
    direct_batch_size = batch_size;

    auto node_weights = direct_node_energy_adjoints;
    Kokkos::parallel_for(
        "Seed direct batch node-energy adjoints",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, batch_size),
        KOKKOS_LAMBDA (const std::size_t structure) {
            const Precision seed = static_cast<Precision>(
                (structure_energies(structure)-reference_energies(structure))
                *residual_scales(structure)
                *residual_scales(structure)/static_cast<double>(batch_size));
            for (std::size_t node=structure_offsets(structure);
                    node<structure_offsets(structure+1); ++node)
                node_weights(node) = seed;
        });

    auto h1_adj = this->H1_adj;
    auto h2_adj = this->H2_adj;
    const int channel_count = num_channels;
    Kokkos::parallel_for(
        "Scale direct readout adjoints by node loss seeds",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, static_cast<std::size_t>(num_nodes)),
        KOKKOS_LAMBDA (const std::size_t node) {
            const Precision weight = node_weights(node);
            for (int channel=0; channel<channel_count; ++channel) {
                h1_adj(node,0,channel) *= weight;
                h2_adj(node,channel) *= weight;
            }
        });
    direct_batch_node_seed_active = true;
    direct_gradient_objective =
        DirectGradientObjective::mean_half_squared_energy;
    direct_gradient_finite_difference_displacement = 0.0;
}

template <typename Precision>
std::vector<std::string> MACEKokkos<Precision>::direct_parameter_names() const
{
    auto names = std::vector<std::string>();
    names.reserve(direct_parameter_gradient_groups.size());
    for (const auto& group : direct_parameter_gradient_groups)
        names.push_back(group.name);
    return names;
}

template <typename Precision>
std::string MACEKokkos<Precision>::direct_parameter_gradient_kernel_policy() const
{
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    return host_memory
        ? "cpu-node-owner-v1" : "device-packed-gemm-v1";
}

template <typename Precision>
std::string MACEKokkos<Precision>::direct_mlp_parameter_gradient_policy() const
{
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    return host_memory ? "cpu-node-owner-v1" : "device-gemm-v1";
}

template <typename Precision>
std::string MACEKokkos<Precision>::direct_h1_parameter_gradient_policy() const
{
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    return host_memory ? "cpu-node-owner-v1" : "device-gemm-v1";
}

template <typename Precision>
std::map<std::string,std::vector<std::size_t>>
MACEKokkos<Precision>::direct_parameter_shapes() const
{
    auto shapes = std::map<std::string,std::vector<std::size_t>>();
    for (const auto& group : direct_parameter_gradient_groups)
        shapes.emplace(group.name, group.shape);
    return shapes;
}

template <typename Precision>
std::map<std::string,std::vector<double>>
MACEKokkos<Precision>::get_direct_parameters() const
{
    if (direct_parameter_gradients_enabled
            && direct_training_parameters.extent(0) != 0) {
        factorized_execution_space.fence("Export direct training parameters");
        const auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), direct_training_parameters);
        auto parameters = std::map<std::string,std::vector<double>>();
        for (const auto& group : direct_parameter_gradient_groups)
            parameters[group.name] = std::vector<double>(
                host.data()+group.offset,
                host.data()+group.offset+group.elements);
        return parameters;
    }
    auto parameters = std::map<std::string,std::vector<double>>();
    for (int lm=0; lm<num_LM; ++lm) {
        const auto coefficients = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), M0_poly_coeff(lm));
        auto& values = parameters[
            "M0_weights.LM"+std::to_string(lm)];
        values.reserve(
            static_cast<std::size_t>(coefficients.extent(0))
            *static_cast<std::size_t>(coefficients.extent(2))
            *M0_term_coefficient_nodes[lm].size());
        for (int type=0; type<coefficients.extent(0); ++type)
            for (int channel=0; channel<coefficients.extent(2); ++channel)
                for (const int node : M0_term_coefficient_nodes[lm])
                    values.push_back(
                        node >= 0 ? coefficients(type,node,channel) : 0.0);
    }
    const auto h1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H1_weights);
    for (int l=0; l<=L_max; ++l) {
        auto& values = parameters["H1_weights.l"+std::to_string(l)];
        values.reserve(static_cast<std::size_t>(num_channels)*num_channels);
        for (int input=0; input<num_channels; ++input)
            for (int output=0; output<num_channels; ++output)
                values.push_back(h1(l,input,output));
    }
    for (int l=0; l<=l_max; ++l) {
        const auto weights = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), A1_weights(l));
        parameters["A1_weights.l"+std::to_string(l)].assign(
            weights.data(), weights.data()+weights.size());
    }
    const auto m1_coefficients = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), M1_poly_coeff);
    {
        auto& values = parameters["M1_weights"];
        values.reserve(m1_coefficients.size());
        for (int type=0; type<m1_coefficients.extent(0); ++type)
            for (int channel=0; channel<m1_coefficients.extent(2); ++channel)
                for (const int node : M1_term_coefficient_nodes)
                    values.push_back(
                        node >= 0
                            ? m1_coefficients(type,node,channel) : 0.0);
    }
    const auto h2_h1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H2_weights_for_H1);
    parameters["H2_weights_for_H1"].assign(
        h2_h1.data(), h2_h1.data()+h2_h1.size());
    const auto h2_m1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), H2_weights_for_M1);
    parameters["H2_weights_for_M1"].assign(
        h2_m1.data(), h2_m1.data()+h2_m1.size());
    const auto readout_1 = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), readout_1_weights);
    parameters["readout_1_weights"].assign(
        readout_1.data(), readout_1.data()+readout_1.size());
    const auto readout_weights = readout_2.get_weights();
    parameters["readout_2.weights.0"] = std::move(readout_weights[0]);
    parameters["readout_2.weights.1"] = std::move(readout_weights[1]);
    return parameters;
}

template <typename Precision>
void MACEKokkos<Precision>::set_direct_parameters(
    const std::map<std::string,std::vector<double>>& parameters)
{
    if (!direct_parameter_gradients_enabled)
        throw std::logic_error(
            "Enable direct parameter gradients before updating parameters.");
    const auto shapes = direct_parameter_shapes();
    for (const auto& [name, values] : parameters) {
        const auto shape = shapes.find(name);
        if (shape == shapes.end()) {
            if (name == "atomic_energies")
                throw std::invalid_argument(
                    "atomic_energies are fixed in easily-trainable-v1.");
            if (name.rfind("compact_radial.", 0) == 0
                    || name == "H0_weights"
                    || name.rfind("A0_weights.", 0) == 0
                    || name.rfind("zbl.", 0) == 0)
                throw std::invalid_argument(
                    "Radial, H0/A0, and ZBL updates require deferred cache "
                    "support; '"+name+"' is fixed in easily-trainable-v1.");
            throw std::invalid_argument(
                "Unknown direct parameter '"+name+"'.");
        }
        const std::size_t expected = checked_element_count(shape->second);
        if (values.size() != expected)
            throw std::invalid_argument(
                "Direct parameter '"+name+"' has the wrong extent.");
        require_finite(values);
    }

    factorized_execution_space.fence("Import direct model parameters");
    auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), direct_training_parameters);
    for (const auto& [name, values] : parameters) {
        const auto& group = direct_parameter_gradient_group(name);
        std::copy(
            values.begin(), values.end(), host.data()+group.offset);
    }
    Kokkos::deep_copy(
        factorized_execution_space, direct_training_parameters, host);
    refresh_direct_execution_weights();
}

template <typename Precision>
void MACEKokkos<Precision>::refresh_direct_execution_weights()
{
    if (direct_training_parameters.extent(0) == 0)
        throw std::logic_error("Direct training parameters are unavailable.");
    const auto parameters = direct_training_parameters;
    const int channels = num_channels;

    for (int lm=0; lm<num_LM; ++lm) {
        const auto& group = direct_parameter_gradient_group(
            "M0_weights.LM"+std::to_string(lm));
        const std::size_t offset = group.offset;
        const std::size_t terms = group.shape[2];
        const std::size_t elements = group.elements;
        const auto weights = M0_weights(lm);
        const auto poly = M0_poly_coeff(lm);
        const auto poly_sources = direct_M0_poly_sources[lm];
        Kokkos::parallel_for(
            "Refresh direct M0 weights",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const std::size_t term = index%terms;
                const std::size_t packed = index/terms;
                const int channel = static_cast<int>(packed%channels);
                const int type = static_cast<int>(packed/channels);
                const Precision value = static_cast<Precision>(
                    parameters(offset+index));
                weights(type,channel,term) = value;
            });
        const std::size_t poly_nodes = poly.extent(1);
        const std::size_t poly_elements =
            static_cast<std::size_t>(poly.extent(0))*poly_nodes*channels;
        Kokkos::parallel_for(
            "Refresh direct M0 polynomial coefficients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, poly_elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int channel = static_cast<int>(index%channels);
                const std::size_t packed = index/channels;
                const std::size_t node = packed%poly_nodes;
                const int type = static_cast<int>(packed/poly_nodes);
                const int source = poly_sources(node);
                poly(type,node,channel) = source >= 0
                    ? static_cast<Precision>(parameters(
                        offset+(static_cast<std::size_t>(type)*channels+channel)
                            *terms+static_cast<std::size_t>(source)))
                    : Precision(0);
            });
        if (standard_m0_module_ready) {
            const auto module = standard_m0_module_weights;
            const auto sources = direct_standard_M0_sources[lm];
            const std::size_t module_terms = sources.extent(0);
            const std::size_t module_elements =
                static_cast<std::size_t>(module.extent(0))*module_terms*channels;
            const int packed_offset = symmetrix::standard_m0::term_offsets[lm];
            Kokkos::parallel_for(
                "Refresh direct standard M0 weights",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, module_elements),
                KOKKOS_LAMBDA (const std::size_t index) {
                    const int channel = static_cast<int>(index%channels);
                    const std::size_t packed = index/channels;
                    const std::size_t term = packed%module_terms;
                    const int type = static_cast<int>(packed/module_terms);
                    const int source = sources(term);
                    const std::size_t source_index =
                        (static_cast<std::size_t>(type)*channels+channel)*terms
                        +static_cast<std::size_t>(source);
                    module(type,packed_offset+term,channel) =
                        static_cast<Precision>(parameters(offset+source_index));
                });
        }
    }

    for (int l=0; l<=L_max; ++l) {
        const auto& group = direct_parameter_gradient_group(
            "H1_weights.l"+std::to_string(l));
        const auto destination = H1_weights;
        const std::size_t offset = group.offset;
        Kokkos::parallel_for(
            "Refresh direct H1 weights",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, group.elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int input = static_cast<int>(index/channels);
                const int output = static_cast<int>(index%channels);
                destination(l,input,output) =
                    static_cast<Precision>(parameters(offset+index));
            });
    }

    for (int l=0; l<=l_max; ++l) {
        const auto& group = direct_parameter_gradient_group(
            "A1_weights.l"+std::to_string(l));
        const auto destination = A1_weights(l);
        const auto transpose = A1_weights_trans(l);
        const std::size_t offset = group.offset;
        Kokkos::parallel_for(
            "Refresh direct A1 weights",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, group.elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int input = static_cast<int>(index/channels);
                const int output = static_cast<int>(index%channels);
                const Precision value = static_cast<Precision>(
                    parameters(offset+index));
                destination(input,output) = value;
                transpose(output,input) = value;
            });
    }

    {
        const auto& group = direct_parameter_gradient_group("M1_weights");
        const std::size_t offset = group.offset;
        const std::size_t terms = group.shape[2];
        const auto weights = M1_weights;
        const auto poly = M1_poly_coeff;
        const auto weight_sources = direct_M1_weight_sources;
        const auto poly_sources = direct_M1_poly_sources;
        const std::size_t weight_elements =
            static_cast<std::size_t>(weights.extent(0))*weights.extent(1)*channels;
        Kokkos::parallel_for(
            "Refresh direct M1 weights",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, weight_elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int channel = static_cast<int>(index%channels);
                const std::size_t packed = index/channels;
                const std::size_t term = packed%weights.extent(1);
                const int type = static_cast<int>(packed/weights.extent(1));
                const std::size_t source = weight_sources(term);
                const std::size_t source_index =
                    (static_cast<std::size_t>(type)*channels+channel)*terms+source;
                weights(type,term,channel) =
                    static_cast<Precision>(parameters(offset+source_index));
            });
        Kokkos::parallel_for(
            "Refresh direct M1 polynomial coefficients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0,
                static_cast<std::size_t>(poly.extent(0))*poly.extent(1)*channels),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int channel = static_cast<int>(index%channels);
                const std::size_t packed = index/channels;
                const std::size_t node = packed%poly.extent(1);
                const int type = static_cast<int>(packed/poly.extent(1));
                const int source = poly_sources(node);
                poly(type,node,channel) = source >= 0
                    ? static_cast<Precision>(parameters(
                        offset+(static_cast<std::size_t>(type)*channels+channel)
                            *terms+static_cast<std::size_t>(source)))
                    : Precision(0);
            });
    }

    const auto refresh_flat = [&] (
        const std::string& name, const auto destination) {
        const auto& group = direct_parameter_gradient_group(name);
        const std::size_t offset = group.offset;
        Kokkos::parallel_for(
            "Refresh direct flat weights",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, group.elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                destination.data()[index] = parameters(offset+index);
            });
    };
    refresh_flat("H2_weights_for_H1", H2_weights_for_H1);
    refresh_flat("H2_weights_for_M1", H2_weights_for_M1);
    {
        const auto& group = direct_parameter_gradient_group("H2_weights_for_H1");
        const auto reverse = H2_weights_for_H1_reverse;
        const std::size_t offset = group.offset;
        if (reverse.data() != nullptr)
            Kokkos::parallel_for(
                "Refresh direct reverse H2 from H1 weights",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, group.elements),
                KOKKOS_LAMBDA (const std::size_t index) {
                    const std::size_t matrix_elements = channels*channels;
                    const std::size_t type = index/matrix_elements;
                    const std::size_t matrix_index = index%matrix_elements;
                    const std::size_t input = matrix_index/channels;
                    const std::size_t output = matrix_index%channels;
                    reverse(type,output*channels+input) =
                        static_cast<H2WeightPrecision>(parameters(offset+index));
                });
    }
    {
        const auto& group = direct_parameter_gradient_group("H2_weights_for_M1");
        const auto reverse = H2_weights_for_M1_reverse;
        const std::size_t offset = group.offset;
        if (reverse.data() != nullptr)
            Kokkos::parallel_for(
                "Refresh direct reverse H2 from M1 weights",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, group.elements),
                KOKKOS_LAMBDA (const std::size_t index) {
                    const std::size_t input = index/channels;
                    const std::size_t output = index%channels;
                    reverse(output*channels+input) =
                        static_cast<H2WeightPrecision>(parameters(offset+index));
                });
    }
    refresh_flat("readout_1_weights", readout_1_weights);

    const auto& first = direct_parameter_gradient_group("readout_2.weights.0");
    const auto& final = direct_parameter_gradient_group("readout_2.weights.1");
    const auto host_parameters = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), direct_training_parameters);
    std::vector<double> first_weights(first.elements);
    std::vector<double> final_weights(final.elements);
    for (std::size_t index=0; index<first.elements; ++index)
        first_weights[index] = host_parameters(first.offset+index);
    for (std::size_t index=0; index<final.elements; ++index)
        final_weights[index] = host_parameters(final.offset+index);
    readout_2.set_weights({std::move(first_weights), std::move(final_weights)});
    factorized_execution_space.fence("Refresh direct execution weights");
    direct_parameter_gradients_ready = false;
}

template <typename Precision>
void MACEKokkos<Precision>::configure_direct_optimizer(
    const std::string& optimizer,
    const double beta1,
    const double beta2,
    const double epsilon,
    const bool amsgrad,
    const double momentum,
    const bool nesterov,
    const double gradient_clip_norm,
    const std::vector<std::vector<std::string>>& parameter_group_names,
    const std::vector<double>& learning_rates,
    const std::vector<double>& weight_decays)
{
    if (!direct_parameter_gradients_enabled)
        throw std::logic_error(
            "Direct optimizer requires direct parameter gradients.");
    if (optimizer != "sgd" && optimizer != "adam" && optimizer != "adamw")
        throw std::invalid_argument("Unsupported direct optimizer '"+optimizer+"'.");
    if (parameter_group_names.empty()
            || parameter_group_names.size() != learning_rates.size()
            || parameter_group_names.size() != weight_decays.size())
        throw std::invalid_argument(
            "Direct optimizer parameter groups are inconsistent.");
    if (!std::isfinite(beta1) || beta1 < 0.0 || beta1 >= 1.0
            || !std::isfinite(beta2) || beta2 < 0.0 || beta2 >= 1.0
            || !std::isfinite(epsilon) || epsilon <= 0.0
            || !std::isfinite(momentum) || momentum < 0.0 || momentum >= 1.0
            || !std::isfinite(gradient_clip_norm) || gradient_clip_norm < 0.0)
        throw std::invalid_argument("Direct optimizer scalar configuration is invalid.");
    if (optimizer == "sgd" && amsgrad)
        throw std::invalid_argument("AMSGrad requires Adam or AdamW.");
    if (optimizer != "sgd" && (momentum != 0.0 || nesterov))
        throw std::invalid_argument("Momentum and Nesterov require SGD.");
    if (nesterov && momentum == 0.0)
        throw std::invalid_argument("Nesterov requires positive momentum.");

    std::vector<int> group_by_parameter(
        direct_training_parameters.extent(0), -1);
    std::vector<unsigned char> assigned(direct_parameter_gradient_groups.size(), 0);
    for (std::size_t optimizer_group=0;
            optimizer_group<parameter_group_names.size(); ++optimizer_group) {
        if (!std::isfinite(learning_rates[optimizer_group])
                || learning_rates[optimizer_group] <= 0.0
                || !std::isfinite(weight_decays[optimizer_group])
                || weight_decays[optimizer_group] < 0.0)
            throw std::invalid_argument(
                "Direct optimizer group scalar is invalid.");
        for (const auto& name : parameter_group_names[optimizer_group]) {
            const auto iterator = std::find_if(
                direct_parameter_gradient_groups.begin(),
                direct_parameter_gradient_groups.end(),
                [&] (const DirectTrainingParameterGroup& group) {
                    return group.name == name;
                });
            if (iterator == direct_parameter_gradient_groups.end())
                throw std::invalid_argument(
                    "Unknown direct optimizer parameter '"+name+"'.");
            const std::size_t group_index = static_cast<std::size_t>(
                iterator-direct_parameter_gradient_groups.begin());
            if (assigned[group_index])
                throw std::invalid_argument(
                    "Direct optimizer parameter is assigned more than once: '"
                    +name+"'.");
            assigned[group_index] = 1;
            std::fill(
                group_by_parameter.begin()+iterator->offset,
                group_by_parameter.begin()+iterator->offset+iterator->elements,
                static_cast<int>(optimizer_group));
        }
    }
    if (std::find(assigned.begin(), assigned.end(), 0) != assigned.end())
        throw std::invalid_argument(
            "Direct optimizer groups do not cover every trainable parameter.");

    direct_optimizer_group_by_parameter = Kokkos::View<int*>(
        "direct_optimizer_group_by_parameter", group_by_parameter.size());
    direct_optimizer_learning_rates = Kokkos::View<double*>(
        "direct_optimizer_learning_rates", learning_rates.size());
    direct_optimizer_weight_decays = Kokkos::View<double*>(
        "direct_optimizer_weight_decays", weight_decays.size());
    auto host_groups = Kokkos::create_mirror_view(
        direct_optimizer_group_by_parameter);
    auto host_rates = Kokkos::create_mirror_view(
        direct_optimizer_learning_rates);
    auto host_decays = Kokkos::create_mirror_view(
        direct_optimizer_weight_decays);
    std::copy(group_by_parameter.begin(), group_by_parameter.end(), host_groups.data());
    std::copy(learning_rates.begin(), learning_rates.end(), host_rates.data());
    std::copy(weight_decays.begin(), weight_decays.end(), host_decays.data());
    Kokkos::deep_copy(direct_optimizer_group_by_parameter, host_groups);
    Kokkos::deep_copy(direct_optimizer_learning_rates, host_rates);
    Kokkos::deep_copy(direct_optimizer_weight_decays, host_decays);

    const std::size_t elements = direct_training_parameters.extent(0);
    direct_optimizer_momentum_buffer = {};
    direct_optimizer_first_moment = {};
    direct_optimizer_second_moment = {};
    direct_optimizer_max_second_moment = {};
    if (optimizer == "sgd" && momentum != 0.0) {
        direct_optimizer_momentum_buffer = Kokkos::View<double*>(
            "direct_optimizer_momentum", elements);
        Kokkos::deep_copy(direct_optimizer_momentum_buffer, 0.0);
    } else if (optimizer != "sgd") {
        direct_optimizer_first_moment = Kokkos::View<double*>(
            "direct_optimizer_first_moment", elements);
        direct_optimizer_second_moment = Kokkos::View<double*>(
            "direct_optimizer_second_moment", elements);
        Kokkos::deep_copy(direct_optimizer_first_moment, 0.0);
        Kokkos::deep_copy(direct_optimizer_second_moment, 0.0);
        if (amsgrad) {
            direct_optimizer_max_second_moment = Kokkos::View<double*>(
                "direct_optimizer_max_second_moment", elements);
            Kokkos::deep_copy(direct_optimizer_max_second_moment, 0.0);
        }
    }
    direct_optimizer_name = optimizer;
    direct_optimizer_beta1 = beta1;
    direct_optimizer_beta2 = beta2;
    direct_optimizer_epsilon = epsilon;
    direct_optimizer_amsgrad = amsgrad;
    direct_optimizer_momentum = momentum;
    direct_optimizer_nesterov = nesterov;
    direct_optimizer_gradient_clip_norm = gradient_clip_norm;
    direct_optimizer_steps = 0;
}

template <typename Precision>
void MACEKokkos<Precision>::set_direct_optimizer_learning_rates(
    const std::vector<double>& learning_rates)
{
    if (learning_rates.size() != direct_optimizer_learning_rates.extent(0))
        throw std::invalid_argument(
            "Direct optimizer learning-rate group count differs.");
    for (const double value : learning_rates)
        if (!std::isfinite(value) || value <= 0.0)
            throw std::invalid_argument(
                "Direct optimizer learning rates must be finite and positive.");
    auto host = Kokkos::create_mirror_view(direct_optimizer_learning_rates);
    std::copy(learning_rates.begin(), learning_rates.end(), host.data());
    Kokkos::deep_copy(direct_optimizer_learning_rates, host);
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::direct_training_workspace_bytes() const
{
    std::size_t bytes = 0;
    const auto add_bytes = [&] (const std::size_t value) {
        if (value > std::numeric_limits<std::size_t>::max()-bytes)
            throw std::length_error("Direct training workspace size overflow.");
        bytes += value;
    };
    const auto add_elements = [&] (
        const std::size_t elements, const std::size_t element_bytes) {
        if (elements != 0
                && element_bytes
                    > std::numeric_limits<std::size_t>::max()/elements)
            throw std::length_error("Direct training workspace size overflow.");
        add_bytes(elements*element_bytes);
    };
    const auto add_view = [&] (const auto& view) {
        using View = std::decay_t<decltype(view)>;
        add_elements(view.size(), sizeof(typename View::non_const_value_type));
    };

    add_view(direct_training_parameters);
    add_view(direct_training_gradients);
    add_view(direct_training_gradient_scratch);
    add_view(direct_training_gradient_accumulator);
    add_view(direct_A1_packed_inputs);
    add_view(direct_A1_packed_adjoints);
    add_view(direct_H1_packed_inputs);
    add_view(direct_H1_packed_adjoints);
    add_view(direct_mlp_packed_inputs);
    add_view(direct_mlp_packed_hidden_derivatives);
    add_view(direct_optimizer_group_by_parameter);
    add_view(direct_optimizer_learning_rates);
    add_view(direct_optimizer_weight_decays);
    add_view(direct_optimizer_momentum_buffer);
    add_view(direct_optimizer_first_moment);
    add_view(direct_optimizer_second_moment);
    add_view(direct_optimizer_max_second_moment);
    for (const auto& view : direct_M0_term_nodes)
        add_view(view);
    for (const auto& view : direct_M0_poly_sources)
        add_view(view);
    for (const auto& view : direct_standard_M0_sources)
        add_view(view);
    add_view(direct_M1_term_nodes);
    add_view(direct_M1_poly_sources);
    add_view(direct_M1_weight_sources);
    add_view(direct_node_energy_adjoints);
    add_view(direct_batch_structure_offsets);
    add_view(direct_batch_reference_energies);
    add_view(direct_batch_energy_residual_scales);
    add_view(direct_batch_energy_values);
    add_view(direct_force_positions);
    add_view(direct_force_cells);
    add_view(direct_force_edge_shifts);
    add_view(direct_force_pbc);
    add_view(direct_force_edge_structures);
    add_view(direct_force_references);
    add_view(direct_force_direction);
    add_view(direct_force_negative_xyz);
    add_view(direct_force_negative_r);
    add_view(direct_force_positive_xyz);
    add_view(direct_force_positive_r);

    // Full-retention training owns both primal and adjoint graph state.
    add_view(Y);
    add_view(Y_grad);
    if (Y_grad_shuffled.data() != Y_grad.data())
        add_view(Y_grad_shuffled);
    add_view(xyz_shuffled);
    add_view(R0);
    add_view(R0_deriv);
    add_view(standard_r0_density_state);
    add_view(R1);
    add_view(R1_deriv);
    add_view(A0);
    add_view(A0_adj);
    add_view(A0_spline_values);
    add_view(A0_spline_derivs);
    add_view(M0);
    add_view(M0_adj);
    add_view(H1);
    add_view(H1_adj);
    add_view(H1_pre_linear_up);
    add_view(Phi1r);
    add_view(dPhi1r);
    add_view(Phi1);
    add_view(dPhi1);
    add_view(A1);
    add_view(A1_adj);
    add_view(A1_spline_values);
    add_view(A1_spline_derivs);
    add_view(M1);
    add_view(M1_adj);
    add_view(H2);
    add_view(H2_adj);
    add_view(node_energies);
    add_view(node_forces);
    add_view(atom_forces);
    add_view(readout_2_output);
    add_bytes(readout_2.workspace_bytes());
    add_bytes(standard_m0_poly_values_capacity_bytes());
    add_bytes(standard_m0_poly_adjoints_capacity_bytes());
    add_bytes(m1_poly_values_capacity_bytes());
    add_bytes(m1_poly_adjoints_capacity_bytes());

    add_view(factorized_workspace_arena);
    add_view(execution_receiver_projection);
    add_view(execution_radial_values);
    add_view(execution_radial_derivatives);
    add_view(execution_coupling_adjoint);
    add_view(execution_source_chunk_offsets);
    add_view(execution_source_edges);
    add_view(execution_direct_source_offsets);
    add_view(execution_direct_source_edges);
    add_view(execution_edge_receivers);
    add_view(streamed_first_neigh);
    add_view(execution_prepared_node_types);
    add_view(execution_prepared_num_neigh);
    add_view(execution_prepared_neigh_indices);
    add_view(execution_prepared_neigh_types);
    add_view(execution_prepared_xyz);
    add_view(execution_prepared_unit_direction);
    add_view(execution_prepared_r);
    add_view(execution_prepared_positions);
    add_view(execution_prepared_reference_positions);
    add_view(execution_prepared_reference_xyz);
    add_view(execution_prepared_displacements);
    add_view(execution_prepared_cell);
    add_view(execution_prepared_inverse_cell);
    add_view(execution_prepared_pbc);
    add_view(execution_prepared_geometry_invalid);
    add_view(execution_prepared_electric_field);
    return bytes;
}

template <typename Precision>
DirectOptimizerStepResult MACEKokkos<Precision>::apply_direct_optimizer_step()
{
    if (!direct_parameter_gradients_ready)
        throw std::logic_error(
            "Direct optimizer requires ready parameter gradients.");
    if (direct_optimizer_name.empty()
            || direct_optimizer_group_by_parameter.extent(0)
                != direct_training_parameters.extent(0))
        throw std::logic_error("Direct optimizer is not configured.");
    const auto gradients = direct_training_gradients;
    double squared_norm = 0.0;
    Kokkos::parallel_reduce(
        "Direct training gradient norm",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index, double& update) {
            const double value = static_cast<double>(gradients(index));
            update += value*value;
        }, squared_norm);
    const double gradient_norm = std::sqrt(squared_norm);
    constexpr bool device_execution = !std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(double);
    if (!std::isfinite(gradient_norm))
        throw std::invalid_argument("Direct training gradient norm is non-finite.");
    double clip_coefficient = 1.0;
    if (direct_optimizer_gradient_clip_norm != 0.0
            && gradient_norm > direct_optimizer_gradient_clip_norm)
        clip_coefficient = direct_optimizer_gradient_clip_norm/gradient_norm;

    if (direct_optimizer_steps == std::numeric_limits<std::size_t>::max())
        throw std::overflow_error("Direct optimizer step count overflow.");
    if (direct_training_workspace_bytes() > direct_parameter_gradients_max_bytes)
        throw std::length_error(
            "Direct training workspace exceeds the configured parameter-gradient "
            "byte limit.");
    const std::size_t next_step = direct_optimizer_steps+1;
    const auto parameters = direct_training_parameters;
    const auto group_ids = direct_optimizer_group_by_parameter;
    const auto learning_rates = direct_optimizer_learning_rates;
    const auto weight_decays = direct_optimizer_weight_decays;
    const std::size_t elements = parameters.extent(0);
    const double momentum_value = direct_optimizer_momentum;
    const bool nesterov = direct_optimizer_nesterov;
    if (direct_optimizer_name == "sgd") {
        const auto momentum_buffer = direct_optimizer_momentum_buffer;
        const std::size_t step = next_step;
        Kokkos::parallel_for(
            "Direct SGD optimizer step",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int group = group_ids(index);
                const double learning_rate = learning_rates(group);
                double gradient = static_cast<double>(gradients(index))
                    *clip_coefficient;
                const double weight_decay = weight_decays(group);
                if (weight_decay != 0.0)
                    gradient += weight_decay*parameters(index);
                if (momentum_value != 0.0) {
                    double buffer = step == 1
                        ? gradient
                        : momentum_value*momentum_buffer(index)+gradient;
                    momentum_buffer(index) = buffer;
                    if (nesterov)
                        gradient += momentum_value*buffer;
                    else
                        gradient = buffer;
                }
                parameters(index) -= learning_rate*gradient;
            });
    } else {
        const bool adamw = direct_optimizer_name == "adamw";
        const auto first_moment = direct_optimizer_first_moment;
        const auto second_moment = direct_optimizer_second_moment;
        const auto max_second_moment = direct_optimizer_max_second_moment;
        const double beta1 = direct_optimizer_beta1;
        const double beta2 = direct_optimizer_beta2;
        const double epsilon = direct_optimizer_epsilon;
        const bool amsgrad = direct_optimizer_amsgrad;
        const double first_correction =
            1.0-std::pow(beta1, static_cast<double>(next_step));
        const double second_correction =
            1.0-std::pow(beta2, static_cast<double>(next_step));
        Kokkos::parallel_for(
            "Direct Adam optimizer step",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, elements),
            KOKKOS_LAMBDA (const std::size_t index) {
                const int group = group_ids(index);
                const double learning_rate = learning_rates(group);
                const double weight_decay = weight_decays(group);
                double parameter = parameters(index);
                double gradient = static_cast<double>(gradients(index))
                    *clip_coefficient;
                if (adamw && weight_decay != 0.0)
                    parameter *= 1.0-learning_rate*weight_decay;
                else if (weight_decay != 0.0)
                    gradient += weight_decay*parameter;
                const double first = beta1*first_moment(index)
                    +(1.0-beta1)*gradient;
                const double second = beta2*second_moment(index)
                    +(1.0-beta2)*gradient*gradient;
                first_moment(index) = first;
                second_moment(index) = second;
                double denominator = second;
                if (amsgrad) {
                    denominator = Kokkos::max(
                        max_second_moment(index), second);
                    max_second_moment(index) = denominator;
                }
                parameter -= learning_rate*(first/first_correction)
                    /(Kokkos::sqrt(denominator/second_correction)+epsilon);
                parameters(index) = parameter;
            });
    }
    refresh_direct_execution_weights();
    direct_optimizer_steps = next_step;
    direct_training_gradient_accumulator_ready = false;
    return {
        gradient_norm, clip_coefficient, clip_coefficient < 1.0,
        next_step};
}

template <typename Precision>
std::size_t MACEKokkos<Precision>::direct_optimizer_step_count() const
{
    return direct_optimizer_steps;
}

template <typename Precision>
std::map<std::string,std::map<std::string,std::vector<double>>>
MACEKokkos<Precision>::direct_optimizer_state() const
{
    factorized_execution_space.fence("Export direct optimizer state");
    std::map<std::string,std::map<std::string,std::vector<double>>> result;
    const auto export_view = [&] (
        const std::string& state_name, const Kokkos::View<double*>& view) {
        if (view.extent(0) == 0)
            return;
        const auto host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(), view);
        for (const auto& group : direct_parameter_gradient_groups)
            result[state_name][group.name] = std::vector<double>(
                host.data()+group.offset,
                host.data()+group.offset+group.elements);
    };
    export_view("momentum", direct_optimizer_momentum_buffer);
    export_view("first_moment", direct_optimizer_first_moment);
    export_view("second_moment", direct_optimizer_second_moment);
    export_view("max_second_moment", direct_optimizer_max_second_moment);
    return result;
}

template <typename Precision>
void MACEKokkos<Precision>::set_direct_optimizer_state(
    const std::size_t step,
    const std::map<std::string,std::map<std::string,std::vector<double>>>& state)
{
    const auto import_view = [&] (
        const std::string& state_name, const Kokkos::View<double*>& view) {
        const auto found_state = state.find(state_name);
        if (view.extent(0) == 0) {
            if (found_state != state.end())
                throw std::invalid_argument(
                    "Unexpected direct optimizer state '"+state_name+"'.");
            return;
        }
        if (found_state == state.end()
                || found_state->second.size()
                    != direct_parameter_gradient_groups.size())
            throw std::invalid_argument(
                "Direct optimizer state is incomplete for '"+state_name+"'.");
        auto host = Kokkos::create_mirror_view(view);
        for (const auto& group : direct_parameter_gradient_groups) {
            const auto found = found_state->second.find(group.name);
            if (found == found_state->second.end()
                    || found->second.size() != group.elements)
                throw std::invalid_argument(
                    "Direct optimizer state extent differs for '"+group.name+"'.");
            require_finite(found->second);
            std::copy(
                found->second.begin(), found->second.end(),
                host.data()+group.offset);
        }
        Kokkos::deep_copy(view, host);
    };
    import_view("momentum", direct_optimizer_momentum_buffer);
    import_view("first_moment", direct_optimizer_first_moment);
    import_view("second_moment", direct_optimizer_second_moment);
    import_view("max_second_moment", direct_optimizer_max_second_moment);
    const std::size_t expected_states =
        (direct_optimizer_momentum_buffer.extent(0) != 0 ? 1 : 0)
        +(direct_optimizer_first_moment.extent(0) != 0 ? 2 : 0)
        +(direct_optimizer_max_second_moment.extent(0) != 0 ? 1 : 0);
    if (state.size() != expected_states)
        throw std::invalid_argument("Direct optimizer state contains unknown buffers.");
    direct_optimizer_steps = step;
}

template <typename Precision>
void MACEKokkos<Precision>::reserve_direct_parameter_gradient_workspace(
    const std::size_t num_nodes,
    const std::size_t reserved_bytes)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    std::size_t additional_elements = 0;
    std::size_t packed_input_elements = 0;
    std::size_t packed_hidden_elements = 0;
    std::size_t packed_A1_input_elements = 0;
    std::size_t packed_A1_adjoint_elements = 0;
    std::size_t packed_H1_input_elements = 0;
    std::size_t packed_H1_adjoint_elements = 0;
    if constexpr (!host_memory) {
        const auto& first = direct_parameter_gradient_group(
            "readout_2.weights.0");
        const std::size_t node_count = num_nodes;
        const std::size_t hidden_width = first.shape.at(0);
        const std::size_t input_width = first.shape.at(1);
        packed_input_elements = checked_element_count({node_count, input_width});
        packed_hidden_elements = checked_element_count({node_count, hidden_width});
        for (int l=0; l<=l_max; ++l) {
            const auto& group = direct_parameter_gradient_group(
                "A1_weights.l"+std::to_string(l));
            const std::size_t rows = checked_element_count({
                num_nodes, static_cast<std::size_t>(2*l+1)});
            packed_A1_input_elements = std::max(
                packed_A1_input_elements,
                checked_element_count({rows, group.shape[0]}));
            packed_A1_adjoint_elements = std::max(
                packed_A1_adjoint_elements,
                checked_element_count({
                    rows, static_cast<std::size_t>(num_channels)}));
        }
        for (int l=0; l<=L_max; ++l) {
            const auto& group = direct_parameter_gradient_group(
                "H1_weights.l"+std::to_string(l));
            const std::size_t rows = checked_element_count({
                num_nodes, static_cast<std::size_t>(2*l+1)});
            packed_H1_input_elements = std::max(
                packed_H1_input_elements,
                checked_element_count({rows, group.shape[0]}));
            packed_H1_adjoint_elements = std::max(
                packed_H1_adjoint_elements,
                checked_element_count({
                    rows, static_cast<std::size_t>(num_channels)}));
        }
        const auto add_capacity_delta = [&] (const std::size_t required,
                                             const std::size_t current) {
            if (required <= current)
                return;
            const std::size_t delta = required-current;
            if (delta > std::numeric_limits<std::size_t>::max()
                    -additional_elements)
                throw std::length_error(
                    "Direct training workspace size overflow.");
            additional_elements += delta;
        };
        add_capacity_delta(
            packed_input_elements, direct_mlp_packed_inputs.extent(0));
        add_capacity_delta(
            packed_hidden_elements,
            direct_mlp_packed_hidden_derivatives.extent(0));
        add_capacity_delta(
            packed_A1_input_elements, direct_A1_packed_inputs.extent(0));
        add_capacity_delta(
            packed_A1_adjoint_elements, direct_A1_packed_adjoints.extent(0));
        add_capacity_delta(
            packed_H1_input_elements, direct_H1_packed_inputs.extent(0));
        add_capacity_delta(
            packed_H1_adjoint_elements, direct_H1_packed_adjoints.extent(0));
    }
    const std::size_t current_workspace = direct_training_workspace_bytes();
    if (current_workspace > direct_parameter_gradients_max_bytes
            || reserved_bytes
                > direct_parameter_gradients_max_bytes-current_workspace
            || additional_elements
                > (direct_parameter_gradients_max_bytes-current_workspace
                    -reserved_bytes)/sizeof(Precision))
        throw std::length_error(
            "Direct training workspace exceeds the configured parameter-gradient byte limit.");
    if constexpr (!host_memory) {
        if (direct_mlp_packed_inputs.extent(0) < packed_input_elements)
            Kokkos::realloc(
                direct_mlp_packed_inputs, packed_input_elements);
        if (direct_mlp_packed_hidden_derivatives.extent(0)
                < packed_hidden_elements)
            Kokkos::realloc(
                direct_mlp_packed_hidden_derivatives,
                packed_hidden_elements);
        if (direct_A1_packed_inputs.extent(0) < packed_A1_input_elements)
            Kokkos::realloc(
                direct_A1_packed_inputs, packed_A1_input_elements);
        if (direct_A1_packed_adjoints.extent(0) < packed_A1_adjoint_elements)
            Kokkos::realloc(
                direct_A1_packed_adjoints, packed_A1_adjoint_elements);
        if (direct_H1_packed_inputs.extent(0) < packed_H1_input_elements)
            Kokkos::realloc(
                direct_H1_packed_inputs, packed_H1_input_elements);
        if (direct_H1_packed_adjoints.extent(0) < packed_H1_adjoint_elements)
            Kokkos::realloc(
                direct_H1_packed_adjoints, packed_H1_adjoint_elements);
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_readout_h2_parameter_gradients(
    const int num_nodes,
    Kokkos::View<const int*> node_types,
    Kokkos::View<const Precision*> node_energy_weights)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    const bool weighted = node_energy_weights.extent(0) != 0;
    if (weighted
            && node_energy_weights.extent(0)
                < static_cast<std::size_t>(num_nodes))
        throw std::invalid_argument(
            "Weighted readout gradients do not cover every graph node.");
    reserve_direct_parameter_gradient_workspace(
        static_cast<std::size_t>(num_nodes));
    const auto gradients = direct_training_gradients;
    const auto h1 = H1;
    const auto h2_adjoint = H2_adj;
    const auto m1 = M1;
    const int channels = num_channels;

    const auto& readout_1 = direct_parameter_gradient_group(
        "readout_1_weights");
    const std::size_t readout_1_offset = readout_1.offset;
    const auto& h2_h1 = direct_parameter_gradient_group(
        "H2_weights_for_H1");
    const std::size_t h2_h1_offset = h2_h1.offset;
    const std::size_t h2_h1_elements = h2_h1.elements;
    const auto& h2_m1 = direct_parameter_gradient_group(
        "H2_weights_for_M1");
    const std::size_t h2_m1_offset = h2_m1.offset;
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if constexpr (!host_memory) {
        using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using TeamMember = TeamPolicy::member_type;
        Kokkos::parallel_for(
            "Direct device readout-1 parameter gradients",
            TeamPolicy(
                factorized_execution_space, channels, Kokkos::AUTO),
            KOKKOS_LAMBDA (const TeamMember& member) {
                const int channel = member.league_rank();
                Precision gradient = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, num_nodes),
                    [=] (const int node, Precision& sum) {
                        sum += (weighted
                                ? node_energy_weights(node) : Precision(1))
                            *static_cast<Precision>(h1(node,0,channel));
                    },
                    gradient);
                Kokkos::single(Kokkos::PerTeam(member), [=] () {
                    gradients(readout_1_offset+channel) = gradient;
                });
            });
        Kokkos::parallel_for(
            "Direct device H2-from-H1 parameter gradients",
            TeamPolicy(
                factorized_execution_space, h2_h1_elements, Kokkos::AUTO),
            KOKKOS_LAMBDA (const TeamMember& member) {
                const std::size_t owner = member.league_rank();
                const int output = static_cast<int>(owner%channels);
                const std::size_t row_owner = owner/channels;
                const int input = static_cast<int>(row_owner%channels);
                const int type = static_cast<int>(row_owner/channels);
                Precision gradient = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, num_nodes),
                    [=] (const int node, Precision& sum) {
                        if (node_types(node) == type)
                            sum += static_cast<Precision>(h1(node,0,input))
                                *static_cast<Precision>(
                                    h2_adjoint(node,output));
                    },
                    gradient);
                Kokkos::single(Kokkos::PerTeam(member), [=] () {
                    gradients(h2_h1_offset+owner) = gradient;
                });
            });
        Kokkos::parallel_for(
            "Direct device H2-from-M1 parameter gradients",
            TeamPolicy(
                factorized_execution_space,
                static_cast<std::size_t>(channels)*channels,
                Kokkos::AUTO),
            KOKKOS_LAMBDA (const TeamMember& member) {
                const std::size_t owner = member.league_rank();
                const int output = static_cast<int>(owner%channels);
                const int input = static_cast<int>(owner/channels);
                Precision gradient = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, num_nodes),
                    [=] (const int node, Precision& sum) {
                        sum += static_cast<Precision>(m1(node,input))
                            *static_cast<Precision>(
                                h2_adjoint(node,output));
                    },
                    gradient);
                Kokkos::single(Kokkos::PerTeam(member), [=] () {
                    gradients(h2_m1_offset+owner) = gradient;
                });
            });
    } else if (Kokkos::DefaultExecutionSpace().concurrency() <= 1) {
        Kokkos::parallel_for(
            "Direct one-worker readout and H2 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, 1),
            KOKKOS_LAMBDA (const int) {
                for (int node=0; node<num_nodes; ++node) {
                    const Precision node_weight = weighted
                        ? node_energy_weights(node) : Precision(1);
                    for (int channel=0; channel<channels; ++channel)
                        gradients(readout_1_offset+channel) += node_weight
                            *static_cast<Precision>(h1(node,0,channel));
                    const std::size_t type = node_types(node);
                    for (int input=0; input<channels; ++input)
                        for (int output=0; output<channels; ++output) {
                            const std::size_t h2_h1_index =
                                (type*static_cast<std::size_t>(channels)+input)
                                    *channels+output;
                            gradients(h2_h1_offset+h2_h1_index) +=
                                static_cast<Precision>(h1(node,0,input))
                                    *static_cast<Precision>(
                                        h2_adjoint(node,output));
                            gradients(
                                h2_m1_offset
                                    +static_cast<std::size_t>(input)*channels
                                    +output) += static_cast<Precision>(
                                        m1(node,input))*static_cast<Precision>(
                                            h2_adjoint(node,output));
                        }
                }
            });
    } else {
        Kokkos::parallel_for(
            "Direct readout-1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, channels),
            KOKKOS_LAMBDA (const int channel) {
                Precision gradient = Precision(0);
                for (int node=0; node<num_nodes; ++node)
                    gradient += (weighted
                            ? node_energy_weights(node) : Precision(1))
                        *static_cast<Precision>(h1(node,0,channel));
                gradients(readout_1_offset+channel) = gradient;
            });
        Kokkos::parallel_for(
            "Direct H2-from-H1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, h2_h1_elements/channels),
            KOKKOS_LAMBDA (const std::size_t owner) {
                const int input = static_cast<int>(owner%channels);
                const int type = static_cast<int>(owner/channels);
                const std::size_t row = h2_h1_offset+owner*channels;
                for (int node=0; node<num_nodes; ++node) {
                    if (node_types(node) != type)
                        continue;
                    const Precision input_value = static_cast<Precision>(
                        h1(node,0,input));
                    for (int output=0; output<channels; ++output)
                        gradients(row+output) +=
                            input_value*static_cast<Precision>(
                                h2_adjoint(node,output));
                }
            });
        Kokkos::parallel_for(
            "Direct H2-from-M1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, channels),
            KOKKOS_LAMBDA (const int input) {
                const std::size_t row = h2_m1_offset
                    +static_cast<std::size_t>(input)*channels;
                for (int node=0; node<num_nodes; ++node) {
                    const Precision input_value = static_cast<Precision>(
                        m1(node,input));
                    for (int output=0; output<channels; ++output)
                        gradients(row+output) +=
                            input_value*static_cast<Precision>(
                                h2_adjoint(node,output));
                }
            });
    }

    const auto& first = direct_parameter_gradient_group("readout_2.weights.0");
    const auto& final = direct_parameter_gradient_group("readout_2.weights.1");
    Kokkos::View<double*> first_gradient(
        "direct_readout_first_gradient", first.elements);
    Kokkos::View<double*> final_gradient(
        "direct_readout_final_gradient", final.elements);
    Kokkos::deep_copy(factorized_execution_space, first_gradient, 0.0);
    Kokkos::deep_copy(factorized_execution_space, final_gradient, 0.0);
    if (weighted) {
        Kokkos::View<double*> weights(
            "direct_readout_node_weights", static_cast<std::size_t>(num_nodes));
        const auto source_weights = node_energy_weights;
        Kokkos::parallel_for(
            "Convert direct readout node weights",
            Kokkos::RangePolicy<decltype(factorized_execution_space),
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0,
                static_cast<std::size_t>(num_nodes)),
            KOKKOS_LAMBDA (const std::size_t node) {
                weights(node) = static_cast<double>(source_weights(node));
            });
        readout_2.accumulate_weight_gradients(
            factorized_execution_space, num_nodes, first_gradient,
            final_gradient, weights, true);
    } else {
        readout_2.accumulate_weight_gradients(
            factorized_execution_space, num_nodes, first_gradient,
            final_gradient, true);
    }
    const auto gradients_view = direct_training_gradients;
    Kokkos::parallel_for(
        "Store direct readout parameter gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, first.elements),
        KOKKOS_LAMBDA (const std::size_t index) {
            gradients_view(first.offset+index) += first_gradient(index);
        });
    Kokkos::parallel_for(
        "Store direct readout output gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, final.elements),
        KOKKOS_LAMBDA (const std::size_t index) {
            gradients_view(final.offset+index) += final_gradient(index);
        });
}

template <typename Precision>
void MACEKokkos<Precision>::compute_M1_parameter_gradients(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    const auto a1 = A1;
    const auto adjoint = M1_adj;
    const auto monomials = M1_monomials;
    const auto gradients = direct_training_gradients;
    const auto& group = direct_parameter_gradient_group("M1_weights");
    const std::size_t offset = group.offset;
    const std::size_t terms = group.shape[2];
    const int channels = num_channels;
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if constexpr (!host_memory) {
        using TeamPolicy = Kokkos::TeamPolicy<Kokkos::DefaultExecutionSpace>;
        using TeamMember = TeamPolicy::member_type;
        Kokkos::parallel_for(
            "Direct device M1 parameter gradients",
            TeamPolicy(
                factorized_execution_space, group.elements, Kokkos::AUTO),
            KOKKOS_LAMBDA (const TeamMember& member) {
                const std::size_t owner = member.league_rank();
                const std::size_t term = owner%terms;
                const std::size_t row_owner = owner/terms;
                const int channel = static_cast<int>(row_owner%channels);
                const int type = static_cast<int>(row_owner/channels);
                Precision gradient = Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(member, num_nodes),
                    [=] (const int node, Precision& sum) {
                        if (node_types(node) != type)
                            return;
                        Precision monomial = Precision(1);
                        for (int variable=0;
                                variable<monomials.extent(1); ++variable) {
                            const int lm = monomials(term,variable);
                            if (lm == -1)
                                break;
                            monomial *= a1(node,lm,channel);
                        }
                        sum += static_cast<Precision>(adjoint(node,channel))
                            *monomial;
                    },
                    gradient);
                Kokkos::single(Kokkos::PerTeam(member), [=] () {
                    gradients(offset+owner) = gradient;
                });
            });
        return;
    }
    if (Kokkos::DefaultExecutionSpace().concurrency() <= 1) {
        Kokkos::parallel_for(
            "Direct one-worker M1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, 1),
            KOKKOS_LAMBDA (const int) {
                for (int node=0; node<num_nodes; ++node)
                    for (std::size_t term=0; term<terms; ++term)
                        for (int channel=0; channel<channels; ++channel) {
                            Precision monomial = Precision(1);
                            for (int variable=0;
                                    variable<monomials.extent(1); ++variable) {
                                const int lm = monomials(term,variable);
                                if (lm == -1)
                                    break;
                                monomial *= a1(node,lm,channel);
                            }
                            const std::size_t index =
                                (static_cast<std::size_t>(node_types(node))*channels
                                    +channel)*terms+term;
                            gradients(offset+index) +=
                                static_cast<Precision>(adjoint(node,channel))
                                    *monomial;
                        }
            });
        return;
    }
    Kokkos::parallel_for(
        "Direct M1 parameter gradients",
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, group.elements/terms),
        KOKKOS_LAMBDA (const std::size_t owner) {
            const int channel = static_cast<int>(owner%channels);
            const int type = static_cast<int>(owner/channels);
            const std::size_t row = offset+owner*terms;
            for (int node=0; node<num_nodes; ++node) {
                if (node_types(node) != type)
                    continue;
                const Precision node_adjoint = static_cast<Precision>(
                    adjoint(node,channel));
                for (std::size_t term=0; term<terms; ++term) {
                    Precision monomial = Precision(1);
                    for (int variable=0; variable<monomials.extent(1);
                         ++variable) {
                        const int lm = monomials(term,variable);
                        if (lm == -1)
                            break;
                        monomial *= a1(node,lm,channel);
                    }
                    gradients(row+term) += node_adjoint*monomial;
                }
            }
        });
}

template <typename Precision>
void MACEKokkos<Precision>::compute_A1_parameter_gradients(
    const int num_nodes)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    const auto phi1 = Phi1;
    const auto adjoint = A1_adj;
    const auto gradients = direct_training_gradients;
    const int channels = num_channels;
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;

    reserve_direct_parameter_gradient_workspace(
        static_cast<std::size_t>(num_nodes));

    for (int l=0; l<=l_max; ++l) {
        const auto& group = direct_parameter_gradient_group(
            "A1_weights.l"+std::to_string(l));
        const int lme_begin = direct_A1_lme_begins.at(
            static_cast<std::size_t>(l));
        const int input_count = static_cast<int>(group.shape[0]);
        const std::size_t offset = group.offset;
        const int components = 2*l+1;
        const int adjoint_lm = l*l;
        const std::size_t node_component_count =
            static_cast<std::size_t>(num_nodes)*components;
        const std::size_t phi_stride_1 = phi1.extent(1);
        const std::size_t phi_stride_2 = phi1.extent(2);
        if constexpr (!host_memory) {
            using UnmanagedMatrix = Kokkos::View<
                Precision**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;
            const std::size_t packed_rows = node_component_count;
            auto packed_input = UnmanagedMatrix(
                direct_A1_packed_inputs.data(), packed_rows, input_count);
            auto packed_adjoint = UnmanagedMatrix(
                direct_A1_packed_adjoints.data(), packed_rows, channels);
            auto gradient = UnmanagedMatrix(
                gradients.data()+offset, input_count, channels);
            const std::size_t packed_columns =
                static_cast<std::size_t>(input_count)+channels;
            Kokkos::parallel_for(
                "Pack direct A1 parameter gradient operands",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0,
                    packed_rows*packed_columns),
                KOKKOS_LAMBDA (const std::size_t flat) {
                    const std::size_t row = flat/packed_columns;
                    const int column = static_cast<int>(flat%packed_columns);
                    const int component = static_cast<int>(
                        row%static_cast<std::size_t>(components));
                    const int node = static_cast<int>(
                        row/static_cast<std::size_t>(components));
                    if (column < input_count) {
                        packed_input(row,column) = static_cast<Precision>(
                            phi1.data()[
                                static_cast<std::size_t>(node)
                                    *phi_stride_1*phi_stride_2
                                +static_cast<std::size_t>(lme_begin)
                                    *phi_stride_2
                                +static_cast<std::size_t>(component)
                                    *input_count
                                +static_cast<std::size_t>(column)]);
                    } else {
                        packed_adjoint(row,column-input_count) =
                            static_cast<Precision>(adjoint(
                                node, adjoint_lm+component,
                                column-input_count));
                    }
                });
            KokkosBlas::gemm(
                factorized_execution_space, "T", "N", Precision(1),
                packed_input, packed_adjoint, Precision(0), gradient);
            continue;
        }
        if (Kokkos::DefaultExecutionSpace().concurrency() <= 1) {
            Kokkos::parallel_for(
                "Direct one-worker A1 parameter gradients",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    factorized_execution_space, 0, 1),
                KOKKOS_LAMBDA (const int) {
                    for (int node=0; node<num_nodes; ++node)
                        for (int component=0; component<components; ++component)
                            for (int input=0; input<input_count; ++input)
                                for (int output=0; output<channels; ++output)
                                    gradients(
                                        offset+static_cast<std::size_t>(input)
                                            *channels+output) +=
                                        static_cast<Precision>(phi1.data()[
                                            static_cast<std::size_t>(node)
                                                *phi_stride_1*phi_stride_2
                                            +static_cast<std::size_t>(lme_begin)
                                                *phi_stride_2
                                            +static_cast<std::size_t>(component)
                                                *input_count
                                            +static_cast<std::size_t>(input)])
                                        *static_cast<Precision>(adjoint(
                                            node,adjoint_lm+component,output));
                });
            continue;
        }
        Kokkos::parallel_for(
            "Direct A1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, input_count),
            KOKKOS_LAMBDA (const int input) {
                const std::size_t row = offset
                    +static_cast<std::size_t>(input)*channels;
                for (int node=0; node<num_nodes; ++node)
                    for (int component=0; component<components; ++component) {
                        const Precision input_value = static_cast<Precision>(
                            phi1.data()[
                            static_cast<std::size_t>(node)*phi_stride_1*phi_stride_2
                            +static_cast<std::size_t>(lme_begin)*phi_stride_2
                            +static_cast<std::size_t>(component)*input_count
                            +static_cast<std::size_t>(input)]);
                        for (int output=0; output<channels; ++output)
                            gradients(row+output) += input_value
                                *static_cast<Precision>(
                                    adjoint(node,adjoint_lm+component,output));
                    }
            });
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_H1_parameter_gradients(
    const int num_nodes)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    const auto m0 = M0;
    const auto adjoint = H1_adj;
    const auto gradients = direct_training_gradients;
    const int channels = num_channels;
    constexpr bool host_memory = std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;

    reserve_direct_parameter_gradient_workspace(
        static_cast<std::size_t>(num_nodes));

    for (int l=0; l<=L_max; ++l) {
        const auto& group = direct_parameter_gradient_group(
            "H1_weights.l"+std::to_string(l));
        const std::size_t offset = group.offset;
        const int components = 2*l+1;
        const int lm_begin = l*l;
        const std::size_t node_component_count =
            static_cast<std::size_t>(num_nodes)*components;
        if constexpr (!host_memory) {
            using UnmanagedMatrix = Kokkos::View<
                Precision**, Kokkos::LayoutRight, Kokkos::MemoryUnmanaged>;
            const int input_count = static_cast<int>(group.shape[0]);
            const std::size_t packed_rows = node_component_count;
            auto packed_input = UnmanagedMatrix(
                direct_H1_packed_inputs.data(), packed_rows, input_count);
            auto packed_adjoint = UnmanagedMatrix(
                direct_H1_packed_adjoints.data(), packed_rows, channels);
            auto gradient = UnmanagedMatrix(
                gradients.data()+offset, input_count, channels);
            const std::size_t packed_columns =
                static_cast<std::size_t>(input_count)+channels;
            const std::size_t packed_elements = checked_element_count({
                packed_rows, packed_columns});
            Kokkos::parallel_for(
                "Pack direct H1 parameter gradient operands",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                    Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, packed_elements),
                KOKKOS_LAMBDA (const std::size_t flat) {
                    const std::size_t row = flat/packed_columns;
                    const int column = static_cast<int>(flat%packed_columns);
                    const int component = static_cast<int>(
                        row%static_cast<std::size_t>(components));
                    const int node = static_cast<int>(
                        row/static_cast<std::size_t>(components));
                    if (column < input_count)
                        packed_input(row,column) = static_cast<Precision>(
                            m0(node,lm_begin+component,column));
                    else
                        packed_adjoint(row,column-input_count) =
                            static_cast<Precision>(adjoint(
                                node,lm_begin+component,column-input_count));
                });
            KokkosBlas::gemm(
                factorized_execution_space, "T", "N", Precision(1),
                packed_input, packed_adjoint, Precision(0), gradient);
            continue;
        }
        if (Kokkos::DefaultExecutionSpace().concurrency() <= 1) {
            Kokkos::parallel_for(
                "Direct one-worker H1 parameter gradients",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    factorized_execution_space, 0, 1),
                KOKKOS_LAMBDA (const int) {
                    for (int node=0; node<num_nodes; ++node)
                        for (int component=0; component<components; ++component)
                            for (int input=0; input<channels; ++input)
                                for (int output=0; output<channels; ++output)
                                    gradients(
                                        offset+static_cast<std::size_t>(input)
                                            *channels+output) +=
                                        static_cast<Precision>(
                                            m0(node,lm_begin+component,input))
                                        *static_cast<Precision>(adjoint(
                                            node,lm_begin+component,output));
                });
            continue;
        }
        Kokkos::parallel_for(
            "Direct H1 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                factorized_execution_space, 0, channels),
            KOKKOS_LAMBDA (const int input) {
                const std::size_t row = offset
                    +static_cast<std::size_t>(input)*channels;
                for (int node=0; node<num_nodes; ++node)
                    for (int component=0; component<components; ++component) {
                        const Precision input_value = static_cast<Precision>(
                            m0(node,lm_begin+component,input));
                        for (int output=0; output<channels; ++output)
                            gradients(row+output) += input_value
                                *static_cast<Precision>(
                                    adjoint(node,lm_begin+component,output));
                    }
            });
    }
}

template <typename Precision>
void MACEKokkos<Precision>::compute_M0_parameter_gradients(
    const int num_nodes,
    Kokkos::View<const int*> node_types)
{
    if (!direct_parameter_gradients_enabled
            || !direct_parameter_gradient_capture_active)
        return;
    const auto a0 = A0;
    const auto adjoint = M0_adj;
    const auto gradients = direct_training_gradients;
    const int channels = num_channels;
    for (int lm=0; lm<num_LM; ++lm) {
        const auto monomials = M0_monomials(lm);
        const auto& group = direct_parameter_gradient_group(
            "M0_weights.LM"+std::to_string(lm));
        const std::size_t offset = group.offset;
        const std::size_t terms = group.shape[2];
        constexpr bool host_memory = std::is_same_v<
            typename Kokkos::DefaultExecutionSpace::memory_space,
            Kokkos::HostSpace>;
        if constexpr (!host_memory) {
            using TeamPolicy = Kokkos::TeamPolicy<
                Kokkos::DefaultExecutionSpace>;
            using TeamMember = TeamPolicy::member_type;
            Kokkos::parallel_for(
                "Direct device M0 parameter gradients",
                TeamPolicy(
                    factorized_execution_space, group.elements, Kokkos::AUTO),
                KOKKOS_LAMBDA (const TeamMember& member) {
                    const std::size_t owner = member.league_rank();
                    const std::size_t term = owner%terms;
                    const std::size_t row_owner = owner/terms;
                    const int channel = static_cast<int>(row_owner%channels);
                    const int type = static_cast<int>(row_owner/channels);
                    Precision gradient = Precision(0);
                    Kokkos::parallel_reduce(
                        Kokkos::TeamThreadRange(member, num_nodes),
                        [=] (const int node, Precision& sum) {
                            if (node_types(node) != type)
                                return;
                            Precision monomial = Precision(1);
                            for (int variable=0;
                                    variable<monomials.extent(1); ++variable) {
                                const int input_lm = monomials(term,variable);
                                if (input_lm == -1)
                                    break;
                                monomial *= a0(node,input_lm,channel);
                            }
                            sum += static_cast<Precision>(
                                adjoint(node,lm,channel))*monomial;
                        },
                        gradient);
                    Kokkos::single(Kokkos::PerTeam(member), [=] () {
                        gradients(offset+owner) = gradient;
                    });
                });
            continue;
        }
        if (Kokkos::DefaultExecutionSpace().concurrency() <= 1) {
            Kokkos::parallel_for(
                "Direct one-worker M0 parameter gradients",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
                    factorized_execution_space, 0, 1),
                KOKKOS_LAMBDA (const int) {
                    for (int node=0; node<num_nodes; ++node)
                        for (std::size_t term=0; term<terms; ++term)
                            for (int channel=0; channel<channels; ++channel) {
                                Precision monomial = Precision(1);
                                for (int variable=0;
                                        variable<monomials.extent(1); ++variable) {
                                    const int input_lm = monomials(term,variable);
                                    if (input_lm == -1)
                                        break;
                                    monomial *= a0(node,input_lm,channel);
                                }
                                const std::size_t index =
                                    (static_cast<std::size_t>(node_types(node))
                                        *channels+channel)*terms+term;
                                gradients(offset+index) += static_cast<Precision>(
                                    adjoint(node,lm,channel))*monomial;
                            }
                });
            continue;
        }
        Kokkos::parallel_for(
            "Direct M0 parameter gradients",
            Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace,
                Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, group.elements/terms),
            KOKKOS_LAMBDA (const std::size_t owner) {
                const int channel = static_cast<int>(owner%channels);
                const int type = static_cast<int>(owner/channels);
                const std::size_t row = offset+owner*terms;
                for (int node=0; node<num_nodes; ++node) {
                    if (node_types(node) != type)
                        continue;
                    const Precision node_adjoint = static_cast<Precision>(
                        adjoint(node,lm,channel));
                    for (std::size_t term=0; term<terms; ++term) {
                        Precision monomial = Precision(1);
                        for (int variable=0; variable<monomials.extent(1);
                             ++variable) {
                            const int input_lm = monomials(term,variable);
                            if (input_lm == -1)
                                break;
                            monomial *= a0(node,input_lm,channel);
                        }
                        gradients(row+term) += node_adjoint*monomial;
                    }
                }
            });
    }
}

template class MACEKokkos<float>;
template class MACEKokkos<double>;
