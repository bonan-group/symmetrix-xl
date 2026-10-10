#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "mace_kokkos.hpp"

namespace {

double elapsed_ms(const std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now()-start).count();
}

bool finite_span(const std::span<const double>& values) {
    return std::all_of(
        values.begin(), values.end(),
        [] (const double value) { return std::isfinite(value); });
}

std::size_t checked_extent_product(
    const std::size_t extent, const std::size_t multiplier,
    const char* const description)
{
    if (extent > std::numeric_limits<std::size_t>::max()/multiplier)
        throw std::invalid_argument(description);
    return extent*multiplier;
}

template <typename Value>
std::size_t prospective_view_capacity(
    const Kokkos::View<Value*>& view,
    const std::size_t required)
{
    if (view.extent(0) >= required)
        return view.extent(0);
    const std::size_t current = view.extent(0);
    const std::size_t doubled =
        current > std::numeric_limits<std::size_t>::max()/2
        ? required : 2*current;
    return std::max(required, std::max(std::size_t(1), doubled));
}

template <typename Value>
void reserve_view_capacity(
    Kokkos::View<Value*>& view,
    const std::size_t required,
    const char* const label)
{
    const std::size_t capacity = prospective_view_capacity(view, required);
    if (capacity == view.extent(0))
        return;
    view = Kokkos::View<Value*>(
        Kokkos::view_alloc(std::string(label), Kokkos::WithoutInitializing),
        capacity);
}

template <typename ExecutionSpace, typename Value>
void upload_span(
    const ExecutionSpace& execution_space,
    const Kokkos::View<Value*>& destination,
    const std::span<const Value> source)
{
    if (source.empty())
        return;
    const auto destination_active = Kokkos::subview(
        destination, Kokkos::make_pair(std::size_t(0), source.size()));
    const auto source_host = Kokkos::View<
        const Value*,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
            source.data(), source.size());
    Kokkos::deep_copy(execution_space, destination_active, source_host);
}

}  // namespace

template <typename Precision>
DirectBatchForceLossResult MACEKokkos<Precision>::compute_prepared_direct_force_loss(
    const std::uint64_t graph_generation,
    const std::span<const double> positions,
    const std::span<const int> edge_shifts,
    const std::span<const double> cells,
    const std::span<const int> pbc,
    const std::span<const std::size_t> structure_offsets,
    const std::span<const int> edge_structures,
    const std::span<const double> reference_forces,
    const double displacement,
    const double neighbor_skin,
    const bool return_forces)
{
    begin_factorized_production_evaluation();
    const bool gradients_were_ready = direct_parameter_gradients_ready;
    bool gradient_transaction_started = false;
    try {
    if (!mace_uses_direct_execution(streamed_edges))
        throw std::invalid_argument(
            "Native direct force-loss training requires streamed_edges='direct'.");
    if (!direct_parameter_gradients_enabled)
        throw std::invalid_argument(
            "Native direct force-loss training requires easy-weight gradients.");
    validate_direct_parameter_profile();
    if (graph_generation == 0
        || graph_generation != factorized_prepared_graph_generation
        || factorized_schedule_dirty)
        throw std::invalid_argument(
            "Native direct force training requires the current Execution graph token.");
    if (execution_prepared_num_feature_nodes
        != static_cast<int>(execution_prepared_node_types.extent(0)))
        throw std::invalid_argument(
            "Native direct force training does not accept distributed graphs.");

    const std::size_t num_nodes = execution_prepared_node_types.extent(0);
    const std::size_t num_edges = execution_prepared_neigh_indices.extent(0);
    if (num_edges > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument(
            "Native direct force training exceeds the supported directed-edge range.");
    if (structure_offsets.empty())
        throw std::invalid_argument(
            "Native direct force-loss geometry and labels have inconsistent extents.");
    const std::size_t batch_size = structure_offsets.size()-1;
    const std::size_t force_components = checked_extent_product(
        num_nodes, 3, "Native direct force-loss force extent overflows.");
    const std::size_t edge_vector_components = checked_extent_product(
        num_edges, 3, "Native direct force-loss edge extent overflows.");
    const std::size_t cell_components = checked_extent_product(
        batch_size, 9, "Native direct force-loss cell extent overflows.");
    const std::size_t pbc_components = checked_extent_product(
        batch_size, 3, "Native direct force-loss PBC extent overflows.");
    if (reference_forces.size() != force_components
        || positions.size() != force_components
        || edge_shifts.size() != edge_vector_components
        || edge_structures.size() != num_edges || cells.size() != cell_components
        || pbc.size() != pbc_components || reference_forces.empty())
        throw std::invalid_argument(
            "Native direct force-loss geometry and labels have inconsistent extents.");
    if (structure_offsets.front() != 0
        || structure_offsets.back() != num_nodes)
        throw std::invalid_argument(
            "Native direct force-loss offsets do not cover the prepared nodes.");
    for (std::size_t index=1; index<structure_offsets.size(); ++index)
        if (structure_offsets[index] <= structure_offsets[index-1])
            throw std::invalid_argument(
                "Native direct force-loss structures must be non-empty and ordered.");
    if (!finite_span(positions) || !finite_span(cells)
        || !finite_span(reference_forces))
        throw std::invalid_argument(
            "Native direct force-loss inputs contain a non-finite value.");
    if (!std::isfinite(displacement) || !(displacement > 0.0))
        throw std::invalid_argument(
            "Native direct force-loss displacement must be finite and positive.");
    if (!std::isfinite(neighbor_skin) || !(neighbor_skin > 0.0)
        || 4.0*displacement > neighbor_skin)
        throw std::invalid_argument(
            "Native direct force-loss displacement exceeds the fixed-graph safety "
            "margin; require 4*displacement <= neighbor_skin.");
    const double cutoff = r_cut;
    if (!std::isfinite(cutoff) || !(cutoff > 0.0))
        throw std::invalid_argument("Native direct force training has an invalid cutoff.");

    for (std::size_t structure=0; structure<batch_size; ++structure) {
        const double* cell = cells.data()+9*structure;
        std::array<int,3> periodic_axes {};
        int periodic_dimensions = 0;
        for (int axis=0; axis<3; ++axis)
            if (pbc[3*structure+axis] != 0)
                periodic_axes[periodic_dimensions++] = axis;
        if (periodic_dimensions == 0)
            continue;
        double gram[3][3] {};
        for (int row=0; row<periodic_dimensions; ++row)
            for (int column=0; column<periodic_dimensions; ++column)
                for (int component=0; component<3; ++component)
                    gram[row][column] +=
                        cell[3*periodic_axes[row]+component]
                        *cell[3*periodic_axes[column]+component];
        double determinant = gram[0][0];
        if (periodic_dimensions == 2) {
            determinant = gram[0][0]*gram[1][1]-gram[0][1]*gram[1][0];
        } else if (periodic_dimensions == 3) {
            determinant =
                gram[0][0]*(gram[1][1]*gram[2][2]-gram[1][2]*gram[2][1])
                -gram[0][1]*(gram[1][0]*gram[2][2]-gram[1][2]*gram[2][0])
                +gram[0][2]*(gram[1][0]*gram[2][1]-gram[1][1]*gram[2][0]);
        }
        if (!std::isfinite(determinant) || !(determinant > 0.0))
            throw std::invalid_argument(
                "Native direct force training requires linearly independent "
                "periodic cell vectors.");
    }

    std::size_t additional_force_bytes = 0;
    const auto add_view_growth = [&] (
        const auto& view, const std::size_t required) {
        using View = std::decay_t<decltype(view)>;
        const std::size_t current = view.extent(0);
        const std::size_t capacity = prospective_view_capacity(view, required);
        const std::size_t growth = capacity-current;
        constexpr std::size_t element_bytes =
            sizeof(typename View::non_const_value_type);
        if (growth > std::numeric_limits<std::size_t>::max()/element_bytes
                || growth*element_bytes
                    > std::numeric_limits<std::size_t>::max()
                        -additional_force_bytes)
            throw std::length_error("Direct force workspace size overflow.");
        additional_force_bytes += growth*element_bytes;
    };
    add_view_growth(direct_force_positions, force_components);
    add_view_growth(direct_force_cells, cell_components);
    add_view_growth(direct_force_edge_shifts, edge_vector_components);
    add_view_growth(direct_force_pbc, pbc_components);
    add_view_growth(direct_force_edge_structures, num_edges);
    add_view_growth(direct_force_references, force_components);
    add_view_growth(direct_force_direction, force_components);
    add_view_growth(direct_force_negative_xyz, edge_vector_components);
    add_view_growth(direct_force_negative_r, num_edges);
    add_view_growth(direct_force_positive_xyz, edge_vector_components);
    add_view_growth(direct_force_positive_r, num_edges);
    add_view_growth(direct_batch_structure_offsets, structure_offsets.size());
    add_view_growth(direct_batch_energy_values, batch_size);
    if (direct_training_gradient_scratch.extent(0)
            != direct_training_gradients.extent(0))
        add_view_growth(
            direct_training_gradient_scratch,
            direct_training_gradients.extent(0));
    reserve_direct_parameter_gradient_workspace(num_nodes, additional_force_bytes);
    reserve_view_capacity(
        direct_force_positions, force_components, "direct_force_positions");
    reserve_view_capacity(
        direct_force_cells, cell_components, "direct_force_cells");
    reserve_view_capacity(
        direct_force_edge_shifts, edge_vector_components,
        "direct_force_edge_shifts");
    reserve_view_capacity(
        direct_force_pbc, pbc_components, "direct_force_pbc");
    reserve_view_capacity(
        direct_force_edge_structures, num_edges,
        "direct_force_edge_structures");
    reserve_view_capacity(
        direct_force_references, force_components, "direct_force_references");
    reserve_view_capacity(
        direct_force_direction, force_components, "direct_force_direction");
    reserve_view_capacity(
        direct_force_negative_xyz, edge_vector_components,
        "direct_force_negative_xyz");
    reserve_view_capacity(
        direct_force_negative_r, num_edges, "direct_force_negative_r");
    reserve_view_capacity(
        direct_force_positive_xyz, edge_vector_components,
        "direct_force_positive_xyz");
    reserve_view_capacity(
        direct_force_positive_r, num_edges, "direct_force_positive_r");
    reserve_view_capacity(
        direct_batch_structure_offsets, structure_offsets.size(),
        "direct_batch_structure_offsets");
    reserve_view_capacity(
        direct_batch_energy_values, batch_size, "direct_batch_energy_values");
    if (direct_training_gradient_scratch.extent(0)
            != direct_training_gradients.extent(0))
        direct_training_gradient_scratch = Kokkos::View<Precision*>(
            "direct_training_gradient_scratch",
            direct_training_gradients.extent(0));
    upload_span(factorized_execution_space, direct_force_positions, positions);
    upload_span(factorized_execution_space, direct_force_cells, cells);
    upload_span(factorized_execution_space, direct_force_edge_shifts, edge_shifts);
    upload_span(factorized_execution_space, direct_force_pbc, pbc);
    upload_span(
        factorized_execution_space, direct_force_edge_structures,
        edge_structures);
    upload_span(
        factorized_execution_space, direct_force_references, reference_forces);
    upload_span(
        factorized_execution_space, direct_batch_structure_offsets,
        structure_offsets);
    constexpr bool device_execution = !std::is_same_v<
        typename Kokkos::DefaultExecutionSpace::memory_space,
        Kokkos::HostSpace>;
    if constexpr (device_execution)
        direct_training_host_to_device_bytes +=
            sizeof(double)*(positions.size()+cells.size()+reference_forces.size())
            +sizeof(int)*(edge_shifts.size()+pbc.size()+edge_structures.size())
            +sizeof(std::size_t)*structure_offsets.size();

    const auto receivers = execution_edge_receivers;
    const auto sources = execution_prepared_neigh_indices;
    const auto edge_structure_values = direct_force_edge_structures;
    const auto offset_values = direct_batch_structure_offsets;
    const auto edge_shift_values = direct_force_edge_shifts;
    const auto pbc_values = direct_force_pbc;
    int invalid_edge_count = 0;
    Kokkos::parallel_reduce(
        "Validate direct force-loss edges",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, num_edges),
        KOKKOS_LAMBDA (const std::size_t edge, int& invalid) {
            const int receiver = receivers(edge);
            const int source = sources(edge);
            const int structure = edge_structure_values(edge);
            if (receiver < 0 || static_cast<std::size_t>(receiver) >= num_nodes
                || source < 0 || static_cast<std::size_t>(source) >= num_nodes
                || structure < 0
                || static_cast<std::size_t>(structure) >= batch_size) {
                invalid += 1;
                return;
            }
            const std::size_t structure_index =
                static_cast<std::size_t>(structure);
            const std::size_t begin = offset_values(structure_index);
            const std::size_t end = offset_values(structure_index+1);
            if (static_cast<std::size_t>(receiver) < begin
                || static_cast<std::size_t>(receiver) >= end
                || static_cast<std::size_t>(source) < begin
                || static_cast<std::size_t>(source) >= end) {
                invalid += 1;
                return;
            }
            for (int axis=0; axis<3; ++axis)
                if (pbc_values(3*structure_index+axis) == 0
                    && edge_shift_values(3*edge+axis) != 0)
                    invalid += 1;
        }, invalid_edge_count);
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(int);
    if (invalid_edge_count != 0)
        throw std::invalid_argument(
            "Native direct force-loss edges are invalid for the prepared batch.");

    const auto position_values = direct_force_positions;
    const auto cell_values = direct_force_cells;
    const auto direction_values = direct_force_direction;
    const auto reconstruct = [&] (
        const double sign,
        const Kokkos::View<double*>& xyz_values,
        const Kokkos::View<double*>& radius_values,
        const char* const label) {
        int invalid_geometry_count = 0;
        Kokkos::parallel_reduce(
            label,
            Kokkos::RangePolicy<decltype(factorized_execution_space),
                Kokkos::IndexType<std::size_t>>(
                    factorized_execution_space, 0, num_edges),
            KOKKOS_LAMBDA (const std::size_t edge, int& invalid) {
                const std::size_t source =
                    static_cast<std::size_t>(sources(edge));
                const std::size_t receiver =
                    static_cast<std::size_t>(receivers(edge));
                const std::size_t structure = static_cast<std::size_t>(
                    edge_structure_values(edge));
                double vector[3];
                double squared_distance = 0.0;
                for (int component=0; component<3; ++component) {
                    const std::size_t source_component = 3*source+component;
                    const std::size_t receiver_component = 3*receiver+component;
                    double source_position = position_values(source_component);
                    double receiver_position = position_values(receiver_component);
                    if (sign != 0.0) {
                        source_position += sign*displacement
                            *direction_values(source_component);
                        receiver_position += sign*displacement
                            *direction_values(receiver_component);
                    }
                    vector[component] = source_position-receiver_position;
                    for (int lattice=0; lattice<3; ++lattice)
                        vector[component] += static_cast<double>(
                            edge_shift_values(3*edge+lattice))
                            *cell_values(9*structure+3*lattice+component);
                    squared_distance += vector[component]*vector[component];
                }
                if (!Kokkos::isfinite(squared_distance)
                    || !(squared_distance > 0.0)) {
                    invalid += 1;
                    for (int component=0; component<3; ++component)
                        xyz_values(3*edge+component) = component == 0 ? cutoff : 0.0;
                    radius_values(edge) = cutoff;
                    return;
                }
                const double distance = Kokkos::sqrt(squared_distance);
                const double scale = distance >= cutoff ? cutoff/distance : 1.0;
                for (int component=0; component<3; ++component)
                    xyz_values(3*edge+component) = scale*vector[component];
                radius_values(edge) = scale*distance;
            }, invalid_geometry_count);
        if constexpr (device_execution)
            direct_training_device_to_host_bytes += sizeof(int);
        if (invalid_geometry_count != 0)
            throw std::invalid_argument(
                "Native direct force training produced a zero or invalid edge.");
    };

    DirectBatchForceLossResult result;
    if (return_forces)
        result.structure_offsets.assign(
            structure_offsets.begin(), structure_offsets.end());
    result.displacement = displacement;
    result.batch_size = batch_size;
    result.num_nodes = num_nodes;
    result.num_edges = num_edges;
    result.num_force_components = force_components;

    const auto base_start = std::chrono::steady_clock::now();
    reconstruct(
        0.0, direct_force_negative_xyz, direct_force_negative_r,
        "Reconstruct direct force-loss base geometry");
    const bool capture_was_active = direct_parameter_gradient_capture_active;
    direct_parameter_gradient_capture_active = false;
    try {
        compute_prepared_factorized_device(
            graph_generation, direct_force_negative_xyz, direct_force_negative_r);
    } catch (...) {
        direct_parameter_gradient_capture_active = capture_was_active;
        throw;
    }
    direct_parameter_gradient_capture_active = capture_was_active;
    reduce_prepared_node_forces(graph_generation);
    const auto energy_values = direct_batch_energy_values;
    const auto node_energy_values = node_energies;
    Kokkos::parallel_for(
        "Reduce direct force-loss structure energies",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, batch_size),
        KOKKOS_LAMBDA (const std::size_t structure) {
            double energy = 0.0;
            for (std::size_t node=offset_values(structure);
                    node<offset_values(structure+1); ++node)
                energy += node_energy_values(node);
            energy_values(structure) = energy;
        });
    const auto energy_values_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),
        Kokkos::subview(
            direct_batch_energy_values,
            Kokkos::make_pair(std::size_t(0), batch_size)));
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(double)*batch_size;
    result.energies.resize(batch_size);
    for (std::size_t structure=0; structure<batch_size; ++structure) {
        const double energy = energy_values_host(structure);
        if (!std::isfinite(energy))
            throw std::invalid_argument(
                "Native direct force training produced a non-finite energy.");
        result.energies[structure] = energy;
    }
    if (return_forces) {
        const auto atom_forces_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),
            Kokkos::subview(
                atom_forces,
                Kokkos::make_pair(std::size_t(0), force_components)));
        result.forces.assign(
            atom_forces_host.data(), atom_forces_host.data()+force_components);
    }
    result.base_evaluation_ms = elapsed_ms(base_start);

    const auto reference_values = direct_force_references;
    const auto force_values = atom_forces;
    int invalid_residual_count = 0;
    Kokkos::parallel_reduce(
        "Form direct force-loss residual direction",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, force_components),
        KOKKOS_LAMBDA (const std::size_t component, int& invalid) {
            const double residual = force_values(component)-reference_values(component);
            direction_values(component) = residual;
            if (!Kokkos::isfinite(residual))
                invalid += 1;
        }, invalid_residual_count);
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(int);
    double squared_residual = 0.0;
    Kokkos::parallel_reduce(
        "Reduce direct force-loss squared residual",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, force_components),
        KOKKOS_LAMBDA (const std::size_t component, double& sum) {
            const double residual = direction_values(component);
            sum += residual*residual;
        }, squared_residual);
    double absolute_residual = 0.0;
    Kokkos::parallel_reduce(
        "Reduce direct force-loss absolute residual",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, force_components),
        KOKKOS_LAMBDA (const std::size_t component, double& sum) {
            sum += Kokkos::abs(direction_values(component));
        }, absolute_residual);
    double max_residual = 0.0;
    Kokkos::parallel_reduce(
        "Reduce direct force-loss maximum residual",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, force_components),
        KOKKOS_LAMBDA (const std::size_t component, double& maximum) {
            const double residual = Kokkos::abs(direction_values(component));
            if (residual > maximum)
                maximum = residual;
        }, Kokkos::Max<double>(max_residual));
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += 3*sizeof(double);
    const double components = static_cast<double>(force_components);
    if (invalid_residual_count != 0 || !std::isfinite(squared_residual)
        || !std::isfinite(absolute_residual)
        || !std::isfinite(max_residual))
        throw std::invalid_argument(
            "Native direct force-loss aggregate is non-finite.");
    result.loss = 0.5*squared_residual/components;
    result.force_rmse = std::sqrt(squared_residual/components);
    result.force_mae = absolute_residual/components;
    result.max_abs_residual = max_residual;
    result.zero_residual = max_residual == 0.0;
    direct_batch_size = batch_size;
    direct_batch_loss = result.loss;
    direct_batch_energies = result.energies;

    if (result.zero_residual) {
        Kokkos::deep_copy(
            factorized_execution_space, direct_training_gradients, 0.0);
        direct_gradient_objective =
            DirectGradientObjective::mean_half_squared_force_finite_difference;
        direct_gradient_finite_difference_displacement = displacement;
        finish_direct_parameter_gradients();
        return result;
    }
    Kokkos::parallel_for(
        "Normalize direct force-loss residual direction",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, force_components),
        KOKKOS_LAMBDA (const std::size_t component) {
            direction_values(component) /= max_residual;
        });

    const auto negative_start = std::chrono::steady_clock::now();
    reconstruct(
        -1.0, direct_force_negative_xyz, direct_force_negative_r,
        "Reconstruct direct force-loss negative geometry");
    compute_prepared_factorized_device(
        graph_generation, direct_force_negative_xyz, direct_force_negative_r);
    gradient_transaction_started = true;
    Kokkos::deep_copy(
        factorized_execution_space, direct_training_gradient_scratch,
        direct_training_gradients);
    result.negative_evaluation_ms = elapsed_ms(negative_start);

    const auto positive_start = std::chrono::steady_clock::now();
    reconstruct(
        1.0, direct_force_positive_xyz, direct_force_positive_r,
        "Reconstruct direct force-loss positive geometry");
    compute_prepared_factorized_device(
        graph_generation, direct_force_positive_xyz, direct_force_positive_r);
    result.positive_evaluation_ms = elapsed_ms(positive_start);

    const auto combination_start = std::chrono::steady_clock::now();
    const double scale_double = -max_residual
        /(2.0*displacement*static_cast<double>(force_components));
    if (!std::isfinite(scale_double)
            || scale_double < static_cast<double>(
                std::numeric_limits<Precision>::lowest())
            || scale_double > static_cast<double>(
                std::numeric_limits<Precision>::max()))
        throw std::invalid_argument(
            "Native direct force-loss gradient scale is not representable.");
    const Precision scale = static_cast<Precision>(scale_double);
    const auto gradients = direct_training_gradients;
    const auto scratch = direct_training_gradient_scratch;
    Kokkos::parallel_for(
        "Combine direct force-loss mixed gradients",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
            factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index) {
            gradients(index) = scale*(gradients(index)-scratch(index));
        });
    double gradient_squared_norm = 0.0;
    Kokkos::parallel_reduce(
        "Reduce direct force-loss gradient norm",
        Kokkos::RangePolicy<decltype(factorized_execution_space),
            Kokkos::IndexType<std::size_t>>(
                factorized_execution_space, 0, gradients.extent(0)),
        KOKKOS_LAMBDA (const std::size_t index, double& sum) {
            const double gradient = static_cast<double>(gradients(index));
            sum += gradient*gradient;
        }, gradient_squared_norm);
    if constexpr (device_execution)
        direct_training_device_to_host_bytes += sizeof(double);
    if (!std::isfinite(std::sqrt(gradient_squared_norm))) {
        Kokkos::deep_copy(
            factorized_execution_space, direct_training_gradients, 0.0);
        direct_parameter_gradients_ready = false;
        direct_gradient_objective = DirectGradientObjective::total_energy;
        direct_gradient_finite_difference_displacement = 0.0;
        throw std::invalid_argument(
            "Native direct force-loss gradient norm is non-finite.");
    }
    direct_batch_size = batch_size;
    direct_batch_loss = result.loss;
    direct_batch_energies = result.energies;
    direct_gradient_objective =
        DirectGradientObjective::mean_half_squared_force_finite_difference;
    direct_gradient_finite_difference_displacement = displacement;
    finish_direct_parameter_gradients();
    result.combination_ms = elapsed_ms(combination_start);
    return result;
    } catch (...) {
        if (gradient_transaction_started || !gradients_were_ready
                || !direct_parameter_gradients_ready)
            invalidate_direct_training_state();
        throw;
    }
}

template DirectBatchForceLossResult
MACEKokkos<double>::compute_prepared_direct_force_loss(
    std::uint64_t,
    std::span<const double>,
    std::span<const int>,
    std::span<const double>,
    std::span<const int>,
    std::span<const std::size_t>,
    std::span<const int>,
    std::span<const double>,
    double,
    double,
    bool);

template DirectBatchForceLossResult
MACEKokkos<float>::compute_prepared_direct_force_loss(
    std::uint64_t,
    std::span<const double>,
    std::span<const int>,
    std::span<const double>,
    std::span<const int>,
    std::span<const std::size_t>,
    std::span<const int>,
    std::span<const double>,
    double,
    double,
    bool);
