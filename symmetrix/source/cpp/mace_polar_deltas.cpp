#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "mace_polar_delta_binding.hpp"
#include "mace_polar_deltas.hpp"
#include "mace_polar_kokkos.hpp"
#include "mace_nonlinear_kokkos.hpp"

// The Python surface of the PolarMACE backbone deltas.  The family's evaluator
// owns the parsed layer (MacePolarKokkosT::polar_deltas); this file binds
// it, block by block and as the composed unrolled chain, so the parity and
// adjoint gates drive the same parsed blocks the family's orchestration does
// rather than a second parse or a re-implementation.
namespace symmetrix::mace_polar {

namespace {

namespace py = pybind11;

using ByteArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

template<typename Chain>
void append_solver_diagnostics(py::dict& result, const Chain& chain)
{
    const auto& decision = chain.solver_decision();
    result["requested_solver"] = std::string(
        execution::long_range::to_string(decision.requested));
    result["resolved_solver"] = std::string(
        execution::long_range::to_string(decision.resolved));
    result["solver_reason"] = decision.reason;
    result["automatic_atom_count"] = chain.automatic_nodes();
    result["automatic_reciprocal_mode_count"] =
        chain.automatic_reciprocal_modes();
    result["automatic_reciprocal_mode_count_is_lower_bound"] =
        chain.automatic_reciprocal_modes_are_lower_bound();
    result["automatic_atom_mode_pair_threshold"] =
        execution::long_range::automatic_pme_atom_mode_pair_threshold;
    if (chain.automatic_reciprocal_modes() != 0
        && chain.automatic_nodes() > std::numeric_limits<std::size_t>::max()
            / chain.automatic_reciprocal_modes())
        throw std::overflow_error(
            "automatic atom-mode diagnostic overflows size_t.");
    result["automatic_atom_mode_pair_count"] = chain.automatic_nodes()
        * chain.automatic_reciprocal_modes();
    result["particle_mesh_shape"] = decision.resolved
            == execution::long_range::LongRangeSolver::pme
        ? chain.effective_pme_mesh()
        : std::array<std::int64_t, 3>{0, 0, 0};
    result["n_k"] = chain.workspace_n_k();
    const auto cache = chain.reciprocal_cache_allocation_usage();
    const auto arena = chain.reciprocal_arena_allocation_usage();
    const auto adjoint = chain.reciprocal_adjoint_allocation_usage();
    const auto total = chain.reciprocal_allocation_usage();
    result["reciprocal_cache_allocation_elements"] = cache.elements;
    result["reciprocal_cache_allocation_bytes"] = cache.bytes;
    result["reciprocal_arena_allocation_elements"] = arena.elements;
    result["reciprocal_arena_allocation_bytes"] = arena.bytes;
    result["reciprocal_adjoint_allocation_elements"] = adjoint.elements;
    result["reciprocal_adjoint_allocation_bytes"] = adjoint.bytes;
    result["reciprocal_allocation_elements"] = total.elements;
    result["reciprocal_allocation_bytes"] = total.bytes;
    // Compatibility aliases: these now report the exact total current-live
    // reciprocal allocation rather than assuming every slot is Precision.
    result["reciprocal_table_elements"] = total.elements;
    result["reciprocal_table_bytes"] = total.bytes;
    const auto realspace_cache = chain.realspace_cache_allocation_usage();
    const auto realspace_arena = chain.realspace_arena_allocation_usage();
    const auto realspace_retained = chain.realspace_retained_allocation_usage();
    result["realspace_cache_allocation_elements"] = realspace_cache.elements;
    result["realspace_cache_allocation_bytes"] = realspace_cache.bytes;
    result["realspace_arena_allocation_elements"] = realspace_arena.elements;
    result["realspace_arena_allocation_bytes"] = realspace_arena.bytes;
    result["realspace_retained_allocation_elements"] =
        realspace_retained.elements;
    result["realspace_retained_allocation_bytes"] = realspace_retained.bytes;
    result["realspace_pair_tile"] = chain.realspace_pair_tile();
}

[[noreturn]] void shape_error(const char* what, const py::array& got)
{
    std::string message = what;
    message += " expects a two-dimensional C-contiguous array, got rank ";
    message += std::to_string(got.ndim());
    message += ".";
    throw std::invalid_argument(message);
}

// The NumPy buffers are host memory while the block views may live in another
// memory space, so every crossing copies rather than aliasing.
template<typename Precision>
Kokkos::View<Precision**, Kokkos::LayoutRight> matrix_from_array(
    const ByteArray& array, const char* what)
{
    if (array.ndim() != 2)
        shape_error(what, array);
    if (array.shape(0) > std::numeric_limits<int>::max()
        || array.shape(1) > std::numeric_limits<int>::max())
        throw std::length_error(
            std::string(what) + " dimensions exceed INT_MAX.");
    const auto rows = static_cast<int>(array.shape(0));
    const auto columns = static_cast<int>(array.shape(1));
    if (rows != 0 && static_cast<std::size_t>(columns)
            > std::numeric_limits<std::size_t>::max()
                / static_cast<std::size_t>(rows))
        throw std::length_error(
            std::string(what) + " element count overflows size_t.");
    const double* data = array.data();
    std::vector<Precision> values(static_cast<std::size_t>(rows) * columns);
    for (std::size_t index = 0; index < values.size(); ++index)
        values[index] = static_cast<Precision>(data[index]);
    Kokkos::View<const Precision**, Kokkos::LayoutRight, Kokkos::HostSpace,
        Kokkos::MemoryTraits<Kokkos::Unmanaged>> host(
            values.data(), rows, columns);
    Kokkos::View<Precision**, Kokkos::LayoutRight> view(
        std::string(what), rows, columns);
    Kokkos::deep_copy(view, host);
    return view;
}

template<typename Precision>
py::array_t<double> array_from_matrix(
    Kokkos::View<const Precision**, Kokkos::LayoutRight> view)
{
    const auto mirror =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    py::array_t<double> array(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(mirror.extent(0)),
        static_cast<py::ssize_t>(mirror.extent(1))});
    double* data = array.mutable_data();
    for (std::size_t row = 0; row < mirror.extent(0); ++row)
        for (std::size_t column = 0; column < mirror.extent(1); ++column)
            data[row * mirror.extent(1) + column] =
                static_cast<double>(mirror(row, column));
    return array;
}

template<typename Precision>
py::array_t<double> array_from_vector(
    const std::vector<Precision>& values, const int rows, const int columns)
{
    py::array_t<double> array(std::vector<py::ssize_t>{rows, columns});
    double* data = array.mutable_data();
    for (std::size_t index = 0; index < values.size(); ++index)
        data[index] = static_cast<double>(values[index]);
    return array;
}

inline py::array_t<double> array_from_values(
    const std::vector<double>& values, const int rows, const int columns)
{
    if (values.size() != static_cast<std::size_t>(rows) * columns)
        throw std::invalid_argument(
            "the polar delta surface produced a buffer of the wrong size.");
    py::array_t<double> array(std::vector<py::ssize_t>{rows, columns});
    std::copy(values.begin(), values.end(), array.mutable_data());
    return array;
}

inline py::array_t<double> array_from_values_1d(
    const std::vector<double>& values)
{
    py::array_t<double> array(values.size());
    std::copy(values.begin(), values.end(), array.mutable_data());
    return array;
}

} // namespace

// The block-level and chain-level surface: one entry point per gate, driven by
// the parsed chain the evaluator owns.
template<typename Evaluator>
void bind_polar_delta_surface(py::class_<Evaluator>& cls)
{
    using Precision = typename Evaluator::precision_type;
    using Chain = PolarDeltaChainKokkosT<Precision>;
    constexpr int bits = sizeof(Precision) == sizeof(double) ? 64 : 32;

    cls.def("polar_delta_ready", [] (const Evaluator& evaluator) {
        return evaluator.polar_delta_ready();
    }, "Whether this evaluator parsed the model's polar block.");

    cls.def("_set_polar_realspace_reverse_strategy",
        [] (Evaluator& evaluator, const std::string& strategy) {
            evaluator.polar_deltas().set_realspace_reverse_strategy(strategy);
        }, py::arg("strategy"),
        "Override the native real-space reverse strategy.");
    cls.def("_set_polar_realspace_materialized_budget_bytes",
        [] (Evaluator& evaluator, const std::size_t bytes) {
            evaluator.polar_deltas().set_realspace_materialized_budget_bytes(
                bytes);
        }, py::arg("bytes"),
        "Set the forced experimental real-space radial-table budget.");

    cls.def("polar_delta_shapes", [bits] (Evaluator& evaluator) {
        Chain& chain = evaluator.polar_deltas();
        py::dict shapes;
        shapes["interaction_count"] = chain.interaction_count();
        shapes["steps"] = chain.steps();
        shapes["charges_dimension"] = chain.charges_dimension();
        shapes["density_width"] = chain.density_width();
        shapes["field_dimension"] = chain.field_dimension();
        shapes["potential_dimension"] = chain.potential_dimension();
        shapes["long_range_scalar_bits"] = chain.long_range_scalar_bits();
        shapes["node_feature_dimension"] =
            chain.field_update().node_feature_dimension();
        shapes["invariant_dimension"] = chain.field_update().source_dimension();
        // The gated readout's own spaces, so a caller can drive the gate and
        // the readout on their real widths rather than a literal.
        shapes["readout_hidden_dimension"] =
            chain.field_update().readout().hidden_dimension();
        shapes["gate_input_dimension"] =
            chain.field_update().readout().gate().input_dimension();
        shapes["gate_output_dimension"] =
            chain.field_update().readout().gate().output_dimension();
        // The chain's own contract: the node-attribute width the caller has to
        // supply (the model's element count) and the per-layer feature width
        // the backbone's retention produces.
        shapes["node_attr_dimension"] = chain.contract().num_elements;
        shapes["dtype_bits"] = bits;
        return shapes;
    }, "The parsed delta layer's dimensions, as the contract declares them.");

    cls.def("polar_delta_workspace", [bits] (Evaluator& evaluator) {
        Chain& chain = evaluator.polar_deltas();
        py::dict result;
        result["nodes"] = chain.workspace_nodes();
        result["graphs"] = chain.workspace_graphs();
        result["n_k"] = chain.workspace_n_k();
        append_solver_diagnostics(result, chain);
        result["precision_bits"] = bits;
        result["execution_space"] = chain.workspace_execution_space();
        result["workspace_replacement_count"] =
            chain.workspace_replacement_count();
        result["workspace_reuse_count"] = chain.workspace_reuse_count();
        result["arena_view_allocation_count"] =
            chain.arena_view_allocation_count();
        result["diagnostic_state_materialization_count"] =
            chain.diagnostic_state_materialization_count();
        result["resident_reverse_count"] = chain.resident_reverse_count();
        result["uvu_workspace_replacement_count"] =
            chain.uvu_workspace_replacement_count();
        const auto graph_outputs = chain.graph_output_diagnostics();
        result["graph_output_family"] = "team_parallel_reductions";
        result["graph_output_quantity_count"] = 8;
        result["graph_output_league_extent"] = graph_outputs.league_extent;
        result["graph_output_node_extent"] = graph_outputs.node_extent;
        result["graph_output_k_extent"] = graph_outputs.k_extent;
        result["graph_output_selected_count"] = graph_outputs.selected_count;
        const auto& realspace_energy =
            chain.realspace_energy_launch_diagnostics();
        result["realspace_energy_selected_count"] =
            realspace_energy.selected_count;
        result["realspace_energy_receiver_replica_league_extent"] =
            realspace_energy.receiver_replica_league_extent;
        result["realspace_energy_graph_reduction_league_extent"] =
            realspace_energy.graph_reduction_league_extent;
        result["realspace_energy_partial_elements"] =
            realspace_energy.partial_elements;
        result["realspace_full_materialized_energy_serial_selected_count"] =
            realspace_energy.full_materialized_serial_selected_count;
        result["realspace_full_materialized_energy_parallel_selected_count"] =
            realspace_energy.full_materialized_parallel_selected_count;
        result["realspace_full_materialized_energy_parallel_edge_extent"] =
            realspace_energy.full_materialized_parallel_edge_extent;
        const auto& realspace_reverse =
            chain.realspace_reverse_launch_diagnostics();
        result["realspace_feature_reverse_selected_count"] =
            realspace_reverse.feature_selected_count;
        result["realspace_feature_reverse_multipole_league_extent"] =
            realspace_reverse.feature_multipole_league_extent;
        result["realspace_feature_reverse_position_league_extent"] =
            realspace_reverse.feature_position_league_extent;
        result["realspace_energy_reverse_selected_count"] =
            realspace_reverse.energy_selected_count;
        result["realspace_energy_reverse_multipole_league_extent"] =
            realspace_reverse.energy_multipole_league_extent;
        result["realspace_energy_reverse_position_league_extent"] =
            realspace_reverse.energy_position_league_extent;
        result["realspace_reverse_strategy"] = std::string(
            execution::long_range::to_string(
                chain.realspace_reverse_strategy()));
        result["realspace_reverse_strategy_requested"] = std::string(
            execution::long_range::to_string(
                chain.realspace_reverse_strategy_request()));
        result["realspace_reverse_strategy_resolved"] = std::string(
            chain.realspace_reverse_strategy_resolved());
        result["realspace_reverse_strategy_reason"] = std::string(
            chain.realspace_reverse_strategy_reason());
        result["realspace_materialized_forward_reducer"] = "segmented";
        result["realspace_feature_owner_selected_count"] =
            realspace_reverse.feature_owner_selected_count;
        result["realspace_energy_owner_selected_count"] =
            realspace_reverse.energy_owner_selected_count;
        result["realspace_feature_pair_atomic_selected_count"] =
            realspace_reverse.feature_pair_atomic_selected_count;
        result["realspace_feature_pair_league_extent"] =
            realspace_reverse.feature_pair_league_extent;
        result["realspace_energy_pair_atomic_selected_count"] =
            realspace_reverse.energy_pair_atomic_selected_count;
        result["realspace_energy_pair_league_extent"] =
            realspace_reverse.energy_pair_league_extent;
        result["realspace_atomic_accumulator_bytes"] =
            realspace_reverse.atomic_accumulator_bytes;
        result["realspace_feature_pair_tile_selected_count"] =
            realspace_reverse.feature_pair_tile_selected_count;
        result["realspace_feature_pair_tile_league_extent"] =
            realspace_reverse.feature_pair_tile_league_extent;
        result["realspace_energy_pair_tile_selected_count"] =
            realspace_reverse.energy_pair_tile_selected_count;
        result["realspace_energy_pair_tile_league_extent"] =
            realspace_reverse.energy_pair_tile_league_extent;
        result["realspace_reverse_tile_width"] =
            realspace_reverse.tile_width;
        result["realspace_reverse_partial_accumulator_bytes"] =
            realspace_reverse.partial_accumulator_bytes;
        result["realspace_feature_materialized_selected_count"] =
            realspace_reverse.feature_materialized_selected_count;
        result["realspace_energy_materialized_selected_count"] =
            realspace_reverse.energy_materialized_selected_count;
        result["realspace_materialized_table_build_count"] =
            realspace_reverse.materialized_table_build_count;
        result["realspace_materialized_table_reuse_count"] =
            realspace_reverse.materialized_table_reuse_count;
        result["realspace_materialized_table_invalidation_count"] =
            realspace_reverse.materialized_table_invalidation_count;
        result["realspace_materialized_admission_rejection_count"] =
            realspace_reverse.materialized_admission_rejection_count;
        result["realspace_materialized_radial_entries"] =
            realspace_reverse.materialized_radial_entries;
        result["realspace_materialized_radial_bytes"] =
            realspace_reverse.materialized_radial_bytes;
        result["realspace_full_materialized_table_build_count"] =
            realspace_reverse.full_materialized_table_build_count;
        result["realspace_full_materialized_table_reuse_count"] =
            realspace_reverse.full_materialized_table_reuse_count;
        result["realspace_full_materialized_table_invalidation_count"] =
            realspace_reverse.full_materialized_table_invalidation_count;
        result["realspace_full_materialized_admission_rejection_count"] =
            realspace_reverse.full_materialized_admission_rejection_count;
        result["realspace_full_materialized_feature_directed_edges"] =
            realspace_reverse.full_materialized_feature_directed_edges;
        result["realspace_full_materialized_energy_directed_edges"] =
            realspace_reverse.full_materialized_energy_directed_edges;
        result["realspace_full_materialized_admission_bytes"] =
            realspace_reverse.full_materialized_admission_bytes;
        result["realspace_full_materialized_retained_bytes"] =
            realspace_reverse.full_materialized_retained_bytes;
        result["realspace_full_materialized_has_gradients"] =
            realspace_reverse.full_materialized_has_gradients;
        result["realspace_full_materialized_feature_serial_selected_count"] =
            realspace_reverse.full_materialized_feature_serial_selected_count;
        result["realspace_full_materialized_feature_atomic_selected_count"] =
            realspace_reverse.full_materialized_feature_atomic_selected_count;
        result["realspace_full_materialized_feature_segmented_selected_count"] =
            realspace_reverse.full_materialized_feature_segmented_selected_count;
        result["realspace_full_materialized_feature_parallel_edge_extent"] =
            realspace_reverse.full_materialized_feature_parallel_edge_extent;
        result["realspace_full_materialized_feature_pair_atomic_reverse_count"] =
            realspace_reverse.full_materialized_feature_pair_atomic_reverse_count;
        result["realspace_full_materialized_energy_pair_atomic_reverse_count"] =
            realspace_reverse.full_materialized_energy_pair_atomic_reverse_count;
        return result;
    }, "Return the extent key and reuse counters of the evaluator-owned polar "
       "workspace.");

    cls.def("polar_e3linear_diagnostics", [] (Evaluator& evaluator) {
        const auto diagnostics=evaluator.polar_deltas().e3linear_diagnostics();
        py::dict result;
        result["linear_count"]=diagnostics.linear_count;
        result["selected_count"]=diagnostics.selected_count;
        result["forward_count"]=diagnostics.forward_count;
        result["reverse_count"]=diagnostics.reverse_count;
        result["fallback_count"]=diagnostics.fallback_count;
        py::dict roles;
        for(const auto& [role,count]:diagnostics.roles) roles[py::str(role)]=count;
        result["roles"]=std::move(roles);
        return result;
    }, "Return validated polar E3Linear selection and launch counters.");

    cls.def("polar_potential_adjoint_diagnostics", [] (Evaluator& evaluator) {
        const auto diagnostics = evaluator.polar_deltas()
            .potential_adjoint_diagnostics();
        py::dict result;
        result["selected_family"] = "node_parallel_staged";
        result["reference_family"] = "graph_serial_reference";
        result["selected_count"] = diagnostics.selected_count;
        result["reference_count"] = diagnostics.reference_count;
        result["selected_node_extent"] = diagnostics.selected_node_extent;
        result["selected_graph_extent"] = diagnostics.selected_graph_extent;
        result["model_extent_bits"] = 31;
        result["flattened_index_bits"] = 64;
        result["selected_stage_count"] = 7;
        result["selected_geometry"] =
            "range64(nodes),range64(graphs),range64(nodes),range64(nodes),"
            "range64(graphs),range64(nodes),range64(graphs)";
        result["selected_kernels"] = py::make_tuple(
            "polar chain external adjoint node contributions",
            "polar chain external adjoint graph reduction",
            "polar chain external adjoint node apply",
            "polar chain correction adjoint node contributions",
            "polar chain correction adjoint graph reduction",
            "polar chain correction adjoint node apply",
            "polar chain correction adjoint strain");
        return result;
    }, "Return the polar potential-adjoint launch family and logical geometry.");

    cls.def("polar_validate_potential_adjoint_extents",
        [] (Evaluator&, const std::uint64_t nodes,
            const std::uint64_t graphs) {
            if (nodes > std::numeric_limits<std::size_t>::max()
                || graphs > std::numeric_limits<std::size_t>::max())
                throw std::length_error(
                    "the polar potential-adjoint extent exceeds size_t.");
            const auto extents = Chain::validate_potential_adjoint_extents(
                static_cast<std::size_t>(nodes),
                static_cast<std::size_t>(graphs));
            py::dict result;
            result["nodes"] = extents.nodes;
            result["graphs"] = extents.graphs;
            result["external_node_elements"] =
                extents.external_node_elements;
            result["external_graph_elements"] =
                extents.external_graph_elements;
            result["correction_node_elements"] =
                extents.correction_node_elements;
            result["correction_graph_elements"] =
                extents.correction_graph_elements;
            result["model_extent_bits"] = 31;
            result["flattened_index_bits"] = 64;
            return result;
        }, py::arg("nodes"), py::arg("graphs"),
        "Validate model and potential-adjoint staging extents without allocation.");

    cls.def("polar_e3linear_microbenchmark", [] (Evaluator& evaluator,
            const int nodes,const int warmups,const int repeats) {
        if(nodes<1||warmups<0||repeats<1)
            throw std::invalid_argument(
                "the polar E3Linear microbenchmark requires positive nodes "
                "and repeats and non-negative warmups.");
        const auto linears=evaluator.polar_deltas().e3linear_modules();
        struct Buffers {
            const E3LinearKokkosT<Precision>* linear;
            Kokkos::View<Precision**,Kokkos::LayoutRight> input,output,seed,
                input_adjoint;
            bool replay_forward;
        };
        std::vector<Buffers> buffers;
        buffers.reserve(linears.size());
        const auto is_replayed=[](const std::string& role) {
            return role=="potential_embedding"||role=="node_embedding"
                ||role=="charge_embedding"||role=="source_embedding"
                ||role=="update_readout_first"
                ||role=="update_readout_mid"
                ||role=="update_readout_final";
        };
        for(const auto* linear:linears) {
            Buffers item{
                linear,
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    "polar E3Linear benchmark input",nodes,
                    linear->input_dimension()),
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    "polar E3Linear benchmark output",nodes,
                    linear->output_dimension()),
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    "polar E3Linear benchmark seed",nodes,
                    linear->output_dimension()),
                Kokkos::View<Precision**,Kokkos::LayoutRight>(
                    "polar E3Linear benchmark input adjoint",nodes,
                    linear->input_dimension()),
                is_replayed(linear->specialization_role())};
            Kokkos::deep_copy(item.input,Precision(0.01));
            Kokkos::deep_copy(item.seed,Precision(0.02));
            buffers.push_back(std::move(item));
        }
        const auto run=[&](const bool specialized) {
            for(auto& item:buffers) {
                if(specialized)
                    item.linear->evaluate(item.input,item.output);
                else item.linear->evaluate_generic(item.input,item.output);
                if(item.replay_forward) {
                    if(specialized)
                        item.linear->evaluate(item.input,item.output);
                    else item.linear->evaluate_generic(item.input,item.output);
                }
                if(specialized)
                    item.linear->reverse(item.seed,item.input_adjoint);
                else item.linear->reverse_generic(item.seed,item.input_adjoint);
            }
        };
        for(int iteration=0;iteration<warmups;++iteration) {
            run(false);
            run(true);
        }
        Kokkos::fence();
        std::vector<double> generic,specialized;
        generic.reserve(repeats);
        specialized.reserve(repeats);
        const auto measure=[&](const bool use_specialized) {
            Kokkos::fence();
            const auto start=std::chrono::steady_clock::now();
            run(use_specialized);
            Kokkos::fence();
            return std::chrono::duration<double,std::milli>(
                std::chrono::steady_clock::now()-start).count();
        };
        for(int iteration=0;iteration<repeats;++iteration) {
            generic.push_back(measure(false));
            specialized.push_back(measure(true));
        }
        const auto median=[](std::vector<double> values) {
            std::sort(values.begin(),values.end());
            const std::size_t middle=values.size()/2;
            return values.size()%2?values[middle]
                :(values[middle-1]+values[middle])/2.0;
        };
        py::dict result;
        result["nodes"]=nodes;
        result["warmups"]=warmups;
        result["repeats"]=repeats;
        result["linear_count"]=linears.size();
        result["forward_launches_per_repeat"]=37;
        result["reverse_launches_per_repeat"]=linears.size();
        result["generic_ms"]=median(generic);
        result["specialized_ms"]=median(specialized);
        result["speedup"]=median(generic)/median(specialized);
        return result;
    },py::arg("nodes"),py::arg("warmups")=3,py::arg("repeats")=7,
       "Benchmark the isolated real polar E3Linear forward/reverse schedule.");

    // --- the individual blocks -------------------------------------------
    // ``name`` names one block and the arrays are its own operands, so each
    // block's forward and reverse are exercisable (and FD-verified) without
    // the chain around them.
    cls.def("polar_delta_block",
        [] (Evaluator& evaluator, const std::string& name, const int layer,
            const std::vector<ByteArray>& inputs) -> py::object {
            Chain& chain = evaluator.polar_deltas();
            auto& update = chain.field_update();
            auto& readout = chain.local_electron_readout();
            if (name == "spin_source_head") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(
                        "spin_source_head takes one input.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "spin_source_head features");
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block head output", features.extent(0),
                    chain.charges_dimension());
                chain.spin_source_heads()[static_cast<std::size_t>(layer)]
                    .evaluate(features, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "feature_mixer") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(
                        "feature_mixer takes one input.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "feature_mixer features");
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block mixer output", features.extent(0),
                    update.node_feature_dimension());
                chain.feature_mixer()[static_cast<std::size_t>(layer)]
                    .evaluate(features, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "fukui_source_map") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(
                        "fukui_source_map takes one input.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "fukui_source_map features");
                chain.fukui_source_map().prepare(
                    static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block fukui output", features.extent(0), 2);
                chain.fukui_source_map().evaluate(features, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "potential_embedding") {
                if (inputs.size() != 3)
                    throw std::invalid_argument(
                        "potential_embedding takes three inputs.");
                auto potential = matrix_from_array<Precision>(
                    inputs[0], "potential_embedding potential");
                auto features = matrix_from_array<Precision>(
                    inputs[1], "potential_embedding features");
                auto charges = matrix_from_array<Precision>(
                    inputs[2], "potential_embedding charges");
                update.embedding().prepare(
                    static_cast<int>(potential.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block embedding output", potential.extent(0),
                    update.embedding().output_dimension());
                update.embedding().evaluate(
                    potential, features, charges, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "dot_products" || name == "tp_out"
                || name == "local_dot_q" || name == "local_dot_v") {
                if (inputs.size() != 2)
                    throw std::invalid_argument(name + " takes two inputs.");
                auto first = matrix_from_array<Precision>(
                    inputs[0], "tensor product input_1");
                auto second = matrix_from_array<Precision>(
                    inputs[1], "tensor product input_2");
                NodeLocalUvuTensorProductKokkosT<Precision>* block = nullptr;
                if (name == "dot_products")
                    block = &update.dot_products();
                else if (name == "tp_out")
                    block = &update.tp_out();
                else if (name == "local_dot_q")
                    block = &readout.dot_products_q();
                else
                    block = &readout.dot_products_v();
                block->prepare(static_cast<int>(first.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block tensor product output", first.extent(0),
                    block->output_dimension());
                block->evaluate(first, second, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "nonlinearity" || name == "local_mlp") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(name + " takes one input.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "mlp input");
                AffineMLPKokkosT<Precision>* block =
                    name == "nonlinearity" ? &update.nonlinearity()
                                           : &readout.mlp();
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block mlp output", input.extent(0),
                    block->output_size());
                block->evaluate(input, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "gate") {
                if (inputs.size() != 1)
                    throw std::invalid_argument("gate takes one input.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "gate input");
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block gate output", input.extent(0),
                    update.readout().gate().output_dimension());
                update.readout().gate().evaluate(input, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "gate_readout") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(
                        "gate_readout takes one input.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "gate_readout features");
                update.readout().prepare(
                    static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block readout output", features.extent(0),
                    update.readout().output_dimension());
                update.readout().evaluate(features, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "local_up_q" || name == "local_up_v") {
                if (inputs.size() != 1)
                    throw std::invalid_argument(name + " takes one input.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "local embedding input");
                auto& block = name == "local_up_q" ? readout.linear_up_q()
                                                   : readout.linear_up_v();
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block local embedding output", input.extent(0),
                    block.output_dimension());
                block.evaluate(input, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "field_update") {
                if (inputs.size() != 4)
                    throw std::invalid_argument(
                        "field_update takes four inputs.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "field_update features");
                auto potential = matrix_from_array<Precision>(
                    inputs[1], "field_update potential");
                auto attrs = matrix_from_array<Precision>(
                    inputs[2], "field_update attrs");
                auto charges = matrix_from_array<Precision>(
                    inputs[3], "field_update charges");
                update.prepare(static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block update output", features.extent(0),
                    update.output_dimension());
                update.evaluate(
                    features, potential, attrs, charges, output);
                return array_from_matrix<Precision>(output);
            }
            if (name == "local_electron_readout") {
                if (inputs.size() != 4)
                    throw std::invalid_argument(
                        "local_electron_readout takes four inputs.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "local readout features");
                auto field = matrix_from_array<Precision>(
                    inputs[1], "local readout field");
                auto initial = matrix_from_array<Precision>(
                    inputs[2], "local readout charges_0");
                auto induced = matrix_from_array<Precision>(
                    inputs[3], "local readout charges_induced");
                readout.prepare(static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block local energy", features.extent(0), 1);
                readout.evaluate(features, field, initial, induced, output);
                return array_from_matrix<Precision>(output);
            }
            throw std::invalid_argument(
                "unknown polar delta block '" + name + "'.");
        },
        py::arg("name"), py::arg("layer"), py::arg("inputs"),
        "Run one parsed delta block on caller-owned rows.");

    cls.def("polar_delta_block_reverse",
        [] (Evaluator& evaluator, const std::string& name, const int layer,
            const std::vector<ByteArray>& inputs,
            const std::vector<ByteArray>& seeds) -> py::object {
            Chain& chain = evaluator.polar_deltas();
            auto& update = chain.field_update();
            auto& readout = chain.local_electron_readout();
            const auto seed_at = [&seeds] (const std::size_t index,
                const char* what) {
                if (index >= seeds.size())
                    throw std::invalid_argument(
                        std::string("the polar delta reverse needs the ") + what
                        + " seed.");
                return seeds[index];
            };
            if (name == "spin_source_head" || name == "feature_mixer") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        name + " takes one input and one seed.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "block features");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "block output seed");
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block feature adjoint", features.extent(0),
                    features.extent(1));
                auto& block = name == "spin_source_head"
                    ? chain.spin_source_heads()[static_cast<std::size_t>(layer)]
                    : chain.feature_mixer()[static_cast<std::size_t>(layer)];
                block.reverse(seed, adjoint, true);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "fukui_source_map") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "fukui_source_map takes one input and one seed.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "fukui features");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "fukui output seed");
                chain.fukui_source_map().prepare(
                    static_cast<int>(features.extent(0)));
                // The block's own reverse reads the intermediates its forward
                // stored, so the forward runs first on the same input.
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block fukui output", features.extent(0), 2);
                chain.fukui_source_map().evaluate(features, output);
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block fukui adjoint", features.extent(0),
                    features.extent(1));
                chain.fukui_source_map().reverse(
                    seed, features, adjoint, true);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "potential_embedding") {
                if (inputs.size() != 3 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "potential_embedding takes three inputs and one seed.");
                auto potential = matrix_from_array<Precision>(
                    inputs[0], "embedding potential");
                auto features = matrix_from_array<Precision>(
                    inputs[1], "embedding features");
                auto charges = matrix_from_array<Precision>(
                    inputs[2], "embedding charges");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "embedding output seed");
                update.embedding().prepare(
                    static_cast<int>(potential.extent(0)));
                auto potential_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block embedding potential adjoint",
                        potential.extent(0), potential.extent(1));
                auto feature_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block embedding feature adjoint",
                        features.extent(0), features.extent(1));
                auto charge_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block embedding charge adjoint",
                        charges.extent(0), charges.extent(1));
                update.embedding().reverse(seed, potential_adjoint,
                    feature_adjoint, charge_adjoint, true);
                py::list result;
                result.append(array_from_matrix<Precision>(potential_adjoint));
                result.append(array_from_matrix<Precision>(feature_adjoint));
                result.append(array_from_matrix<Precision>(charge_adjoint));
                return result;
            }
            if (name == "dot_products" || name == "tp_out"
                || name == "local_dot_q" || name == "local_dot_v") {
                if (inputs.size() != 2 || seeds.size() != 1)
                    throw std::invalid_argument(
                        name + " takes two inputs and one seed.");
                auto first = matrix_from_array<Precision>(
                    inputs[0], "tensor product input_1");
                auto second = matrix_from_array<Precision>(
                    inputs[1], "tensor product input_2");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "tensor product output seed");
                NodeLocalUvuTensorProductKokkosT<Precision>* block = nullptr;
                if (name == "dot_products")
                    block = &update.dot_products();
                else if (name == "tp_out")
                    block = &update.tp_out();
                else if (name == "local_dot_q")
                    block = &readout.dot_products_q();
                else
                    block = &readout.dot_products_v();
                block->prepare(static_cast<int>(first.extent(0)));
                auto first_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block tensor product input_1 adjoint",
                        first.extent(0), first.extent(1));
                auto second_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block tensor product input_2 adjoint",
                        second.extent(0), second.extent(1));
                block->reverse(
                    seed, first, second, first_adjoint, second_adjoint, true);
                py::list result;
                result.append(array_from_matrix<Precision>(first_adjoint));
                result.append(array_from_matrix<Precision>(second_adjoint));
                return result;
            }
            if (name == "nonlinearity" || name == "local_mlp") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        name + " takes one input and one seed.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "mlp input");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "mlp output seed");
                AffineMLPKokkosT<Precision>* block =
                    name == "nonlinearity" ? &update.nonlinearity()
                                           : &readout.mlp();
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block mlp output", input.extent(0),
                    block->output_size());
                block->evaluate(input, output);
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block mlp adjoint", input.extent(0),
                    input.extent(1));
                block->reverse(input, seed, adjoint);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "gate") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "gate takes one input and one seed.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "gate input");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "gate output seed");
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block gate output", input.extent(0),
                    update.readout().gate().output_dimension());
                update.readout().gate().evaluate(input, output);
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block gate adjoint", input.extent(0),
                    input.extent(1));
                update.readout().gate().reverse(seed, input, adjoint, true);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "gate_readout") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "gate_readout takes one input and one seed.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "gate readout features");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "gate readout output seed");
                update.readout().prepare(
                    static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block readout output", features.extent(0),
                    update.readout().output_dimension());
                update.readout().evaluate(features, output);
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block readout adjoint", features.extent(0),
                    features.extent(1));
                update.readout().reverse(seed, features, adjoint, true);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "local_up_q" || name == "local_up_v") {
                if (inputs.size() != 1 || seeds.size() != 1)
                    throw std::invalid_argument(
                        name + " takes one input and one seed.");
                auto input = matrix_from_array<Precision>(
                    inputs[0], "local embedding input");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "local embedding output seed");
                auto& block = name == "local_up_q" ? readout.linear_up_q()
                                                   : readout.linear_up_v();
                auto adjoint = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block local embedding adjoint", input.extent(0),
                    input.extent(1));
                block.reverse(seed, adjoint, true);
                return array_from_matrix<Precision>(adjoint);
            }
            if (name == "field_update") {
                if (inputs.size() != 4 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "field_update takes four inputs and one seed.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "field_update features");
                auto potential = matrix_from_array<Precision>(
                    inputs[1], "field_update potential");
                auto attrs = matrix_from_array<Precision>(
                    inputs[2], "field_update attrs");
                auto charges = matrix_from_array<Precision>(
                    inputs[3], "field_update charges");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "field_update output seed");
                update.prepare(static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block update output", features.extent(0),
                    update.output_dimension());
                update.evaluate(features, potential, attrs, charges, output);
                auto feature_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block update feature adjoint",
                        features.extent(0), features.extent(1));
                auto potential_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block update potential adjoint",
                        potential.extent(0), potential.extent(1));
                auto attrs_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block update attrs adjoint", attrs.extent(0),
                        attrs.extent(1));
                auto charges_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block update charge adjoint", charges.extent(0),
                        charges.extent(1));
                update.reverse(seed, features, potential, attrs, charges,
                    feature_adjoint, potential_adjoint, attrs_adjoint,
                    charges_adjoint, true);
                py::list result;
                result.append(array_from_matrix<Precision>(feature_adjoint));
                result.append(array_from_matrix<Precision>(potential_adjoint));
                result.append(array_from_matrix<Precision>(attrs_adjoint));
                result.append(array_from_matrix<Precision>(charges_adjoint));
                return result;
            }
            if (name == "local_electron_readout") {
                if (inputs.size() != 4 || seeds.size() != 1)
                    throw std::invalid_argument(
                        "local_electron_readout takes four inputs and one "
                        "seed.");
                auto features = matrix_from_array<Precision>(
                    inputs[0], "local readout features");
                auto field = matrix_from_array<Precision>(
                    inputs[1], "local readout field");
                auto initial = matrix_from_array<Precision>(
                    inputs[2], "local readout charges_0");
                auto induced = matrix_from_array<Precision>(
                    inputs[3], "local readout charges_induced");
                auto seed = matrix_from_array<Precision>(seed_at(0, "output"),
                    "local readout output seed");
                readout.prepare(static_cast<int>(features.extent(0)));
                auto output = Kokkos::View<Precision**, Kokkos::LayoutRight>(
                    "polar block local energy", features.extent(0), 1);
                readout.evaluate(features, field, initial, induced, output);
                auto feature_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block local feature adjoint",
                        features.extent(0), features.extent(1));
                auto field_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block local field adjoint", field.extent(0),
                        field.extent(1));
                auto initial_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block local initial adjoint", initial.extent(0),
                        initial.extent(1));
                auto induced_adjoint =
                    Kokkos::View<Precision**, Kokkos::LayoutRight>(
                        "polar block local induced adjoint", induced.extent(0),
                        induced.extent(1));
                readout.reverse(seed, features, field, initial, induced,
                    feature_adjoint, field_adjoint, initial_adjoint,
                    induced_adjoint, true);
                py::list result;
                result.append(array_from_matrix<Precision>(feature_adjoint));
                result.append(array_from_matrix<Precision>(field_adjoint));
                result.append(array_from_matrix<Precision>(initial_adjoint));
                result.append(array_from_matrix<Precision>(induced_adjoint));
                return std::move(result);
            }
            throw std::invalid_argument(
                "unknown polar delta block '" + name + "'.");
        },
        py::arg("name"), py::arg("layer"), py::arg("inputs"), py::arg("seeds"),
        "Run one parsed delta block's reverse on caller-owned rows.");

    // --- the composed chain ----------------------------------------------
    cls.def("polar_delta_chain",
        [] (Evaluator& evaluator, const ByteArray& node_features,
            const ByteArray& node_attrs, const ByteArray& positions,
            const ByteArray& cell, const ByteArray& graph_pbc,
            const ByteArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential,
            const std::string& polar_electrostatics) {
            Chain& chain = evaluator.polar_deltas();
            typename Chain::Inputs inputs;
            inputs.solver_request = execution::long_range::
                parse_long_range_solver_request(polar_electrostatics);
            const auto row_major = [] (const ByteArray& array,
                std::vector<double>& destination) {
                destination.assign(array.data(), array.data() + array.size());
            };
            row_major(node_features, inputs.node_features);
            row_major(node_attrs, inputs.node_attrs);
            row_major(positions, inputs.positions);
            row_major(cell, inputs.cell);
            row_major(external_potential, inputs.external_potential);
            {
                const auto* index_data = node_graph_index.data();
                for (py::ssize_t index = 0; index < node_graph_index.size();
                    ++index)
                    inputs.node_graph_index.push_back(
                        static_cast<int>(index_data[index]));
                const auto* pbc_data = graph_pbc.data();
                for (py::ssize_t index = 0; index < graph_pbc.size(); ++index)
                    inputs.graph_pbc.push_back(
                        static_cast<int>(pbc_data[index]));
            }
            inputs.total_charge = total_charge;
            inputs.total_spin = total_spin;
            const auto state = chain.forward(inputs);
            const int nodes = static_cast<int>(state.nodes);
            py::dict result;
            result["spin_charge_density"] = array_from_values(
                state.spin_charge_density, nodes, chain.charges_dimension());
            result["charge_density"] = array_from_values(
                state.charge_density, nodes, chain.density_width());
            result["spin_density"] = array_from_values(
                state.spin_density, nodes, chain.density_width());
            result["pre_loop_density"] = array_from_values(
                state.pre_loop_density, nodes, chain.charges_dimension());
            result["local_electron_energy"] = array_from_values(
                state.local_electron_energy, nodes, 1);
            result["features_mixed"] = array_from_values(
                state.features_mixed, nodes,
                chain.field_update().node_feature_dimension());
            py::list features;
            for (const auto& step : state.potential_features)
                features.append(array_from_values(step, nodes,
                    chain.potential_dimension()));
            result["potential_features"] = features;
            py::list fukui;
            for (const auto& step : state.fukui_sources)
                fukui.append(array_from_values(step, nodes, 2));
            result["fukui_sources"] = fukui;
            py::list history;
            for (const auto& step : state.density_history)
                history.append(array_from_values(
                    step, nodes, chain.charges_dimension()));
            result["density_history"] = history;
            return result;
        },
        py::arg("node_features"), py::arg("node_attrs"), py::arg("positions"),
        py::arg("cell"), py::arg("graph_pbc"), py::arg("node_graph_index"),
        py::arg("total_charge"), py::arg("total_spin"),
        py::arg("external_potential"),
        py::arg("polar_electrostatics") = "direct_reciprocal",
        "Drive the unrolled polar chain and return its state.");

    cls.def("polar_delta_composed_forward",
        [] (Evaluator& evaluator, const ByteArray& node_features,
            const ByteArray& node_attrs, const ByteArray& positions,
            const ByteArray& cell, const ByteArray& graph_pbc,
            const ByteArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential,
            const bool compute_reverse,
            const std::string& polar_electrostatics) {
            Chain& chain = evaluator.polar_deltas();
            typename Chain::Inputs inputs;
            inputs.solver_request = execution::long_range::
                parse_long_range_solver_request(polar_electrostatics);
            const auto row_major = [] (const ByteArray& array,
                std::vector<double>& destination) {
                destination.assign(array.data(), array.data() + array.size());
            };
            row_major(node_features, inputs.node_features);
            row_major(node_attrs, inputs.node_attrs);
            row_major(positions, inputs.positions);
            row_major(cell, inputs.cell);
            row_major(external_potential, inputs.external_potential);
            for (py::ssize_t index = 0; index < node_graph_index.size();
                ++index)
                inputs.node_graph_index.push_back(
                    static_cast<int>(node_graph_index.data()[index]));
            for (py::ssize_t index = 0; index < graph_pbc.size(); ++index)
                inputs.graph_pbc.push_back(
                    static_cast<int>(graph_pbc.data()[index]));
            inputs.total_charge = total_charge;
            inputs.total_spin = total_spin;

            chain.set_realspace_full_materialized_gradients_required(
                compute_reverse);
            const auto composed = chain.forward_composed_energy(
                inputs, /*materialize_public_state=*/false);
            const auto& state = composed.state;
            const int nodes = static_cast<int>(state.nodes);
            const int graphs = static_cast<int>(state.graphs);
            py::dict result;
            result["energy"] = array_from_values(composed.energy, graphs, 1);
            result["local_electron_energy"] = array_from_values(
                composed.local_electron_energy, graphs, 1);
            result["electrostatic_energy_kspace"] = array_from_values(
                composed.electrostatic_energy, graphs, 1);
            result["electrostatic_energy_correction"] = array_from_values(
                composed.correction_energy, graphs, 1);
            result["external_field_energy"] = array_from_values(
                composed.external_field_energy, graphs, 1);
            result["dipole"] =
                array_from_values(composed.dipole, graphs, 3);
            if (compute_reverse) {
                const auto reverse =
                    chain.reverse_composed_energy(inputs, state);
                const auto& adjoints = reverse.adjoints;
                const int layers = chain.interaction_count();
                const int width =
                    chain.field_update().node_feature_dimension();
                result["node_features_adjoint"] = array_from_values(
                    adjoints.node_features, nodes, layers * width);
                result["node_attrs_adjoint"] = array_from_values(
                    adjoints.node_attrs, nodes, chain.field_update()
                        .source_embedding().input_dimension());
                result["positions_adjoint"] = array_from_values(
                    adjoints.positions, nodes, 3);
                result["external_potential_adjoint"] = array_from_values(
                    adjoints.external_potential,
                    static_cast<int>(adjoints.external_potential.size() / 4),
                    4);
                result["total_charge_adjoint"] = array_from_values(
                    adjoints.total_charge,
                    static_cast<int>(adjoints.total_charge.size()), 1);
                result["total_spin_adjoint"] = array_from_values(
                    adjoints.total_spin,
                    static_cast<int>(adjoints.total_spin.size()), 1);
                result["strain_adjoint"] = array_from_values(
                    adjoints.strain,
                    static_cast<int>(adjoints.total_charge.size()), 9);
                result["stress"] = array_from_values(reverse.stress,
                    static_cast<int>(adjoints.total_charge.size()), 9);
            }
            return result;
        },
        py::arg("node_features"), py::arg("node_attrs"),
        py::arg("positions"), py::arg("cell"), py::arg("graph_pbc"),
        py::arg("node_graph_index"), py::arg("total_charge"),
        py::arg("total_spin"), py::arg("external_potential"),
        py::arg("compute_reverse") = false,
        py::arg("polar_electrostatics") = "direct_reciprocal",
        "Drive the production polar forward composition entirely in native "
        "code and return only compact state and graph outputs.  When "
        "compute_reverse is true, the same native forward state immediately "
        "feeds the complete reverse without a repeated forward pass.");

    cls.def("_evaluate_polar_prepared",
        [] (Evaluator& evaluator, const std::uint64_t graph_generation,
            const IntArray& node_types, const ByteArray& positions,
            const ByteArray& cell, const IntArray& graph_pbc,
            const IntArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential,
            const bool include_forces, const bool include_stress,
            const bool include_dipole,
            const std::string& polar_electrostatics,
            const std::array<std::int64_t,3>& pme_mesh,
            const std::int64_t pme_spline_order) {
            evaluator.clear_layer_product_seed();
            if(total_charge.size()!=1||total_spin.size()!=1
                ||cell.size()!=9||graph_pbc.size()!=3
                ||external_potential.size()!=4)
                throw std::invalid_argument(
                    "polar whole-model evaluation supports exactly one input "
                    "system; mixed or multi-system batches are unsupported.");
            PreparedEvaluationRequest request;
            request.graph_generation=graph_generation;
            request.node_types=std::span<const int>(
                node_types.data(),node_types.size());
            request.positions=std::span<const double>(
                positions.data(),positions.size());
            request.cell=std::span<const double>(cell.data(),cell.size());
            request.graph_pbc=std::span<const int>(
                graph_pbc.data(),graph_pbc.size());
            request.node_graph_index=std::span<const int>(
                node_graph_index.data(),node_graph_index.size());
            request.total_charge=total_charge.front();
            request.total_spin=total_spin.front();
            std::copy(external_potential.data(),
                external_potential.data()+external_potential.size(),
                request.external_potential.begin());
            request.include_forces=include_forces;
            request.include_stress=include_stress;
            request.include_dipole=include_dipole;
            request.solver_request=execution::long_range::
                parse_long_range_solver_request(polar_electrostatics);
            request.pme_mesh=pme_mesh;
            request.pme_spline_order=pme_spline_order;

            const auto evaluation=evaluator.evaluate_prepared(request);
            py::dict result;
            result["energy"]=evaluation.energy;
            result["free_energy"]=evaluation.free_energy;
            if(include_forces)
                result["forces"]=array_from_values(
                    evaluation.forces,
                    static_cast<int>(node_types.size()),3);
            if(include_stress)
                result["stress"]=array_from_values_1d(evaluation.stress);
            if(include_dipole)
                result["dipole"]=array_from_values_1d(evaluation.dipole);
            const auto& diagnostics=evaluation.diagnostics;
            result["requested_solver"]=std::string(
                execution::long_range::to_string(
                    diagnostics.requested_solver));
            result["resolved_solver"]=std::string(
                execution::long_range::to_string(
                    diagnostics.resolved_solver));
            result["solver_reason"]=diagnostics.solver_reason;
            result["automatic_atom_count"]=diagnostics.automatic_atom_count;
            result["automatic_reciprocal_mode_count"]=
                diagnostics.automatic_reciprocal_mode_count;
            result["automatic_reciprocal_mode_count_is_lower_bound"]=
                diagnostics.automatic_reciprocal_mode_count_is_lower_bound;
            result["automatic_atom_mode_pair_count"]=
                diagnostics.automatic_atom_mode_pair_count;
            result["automatic_atom_mode_pair_threshold"]=
                diagnostics.automatic_atom_mode_pair_threshold;
            result["n_k"]=diagnostics.n_k;
            result["reciprocal_allocation_elements"]=
                diagnostics.reciprocal_allocation_elements;
            result["reciprocal_allocation_bytes"]=
                diagnostics.reciprocal_allocation_bytes;
            result["reciprocal_table_elements"]=
                diagnostics.reciprocal_table_elements;
            result["reciprocal_table_bytes"]=
                diagnostics.reciprocal_allocation_bytes;
            result["particle_mesh_active_bytes"]=
                diagnostics.particle_mesh_active_bytes;
            result["particle_mesh_capacity_bytes"]=
                diagnostics.particle_mesh_capacity_bytes;
            result["particle_mesh_volume"]=diagnostics.particle_mesh_volume;
            result["particle_mesh_shape"]=diagnostics.particle_mesh_shape;
            result["particle_mesh_rebuilds"]=diagnostics.particle_mesh_rebuilds;
            result["particle_mesh_reuses"]=diagnostics.particle_mesh_reuses;
            result["particle_mesh_forward_plan_builds"]=
                diagnostics.particle_mesh_forward_plan_builds;
            result["particle_mesh_forward_plan_reuses"]=
                diagnostics.particle_mesh_forward_plan_reuses;
            result["particle_mesh_inverse_plan_builds"]=
                diagnostics.particle_mesh_inverse_plan_builds;
            result["particle_mesh_inverse_plan_reuses"]=
                diagnostics.particle_mesh_inverse_plan_reuses;
            result["backbone_fallback_count"]=
                diagnostics.backbone_fallback_count;
            result["polar_fallback_count"]=
                diagnostics.polar_fallback_count;
            result["fallback_count"]=diagnostics.fallback_count;
            result["reverse_strategy"]=diagnostics.reverse_strategy;
            result["cumulative_transaction_count"]=
                diagnostics.cumulative_transaction_count;
            result["cumulative_polar_reverse_count"]=
                diagnostics.cumulative_polar_reverse_count;
            result["backbone_evaluation_count"]=
                diagnostics.backbone_evaluation_count;
            result["backbone_forward_layer_count"]=
                diagnostics.backbone_forward_layer_count;
            result["backbone_reverse_layer_count"]=
                diagnostics.backbone_reverse_layer_count;
            result["seeded_backbone_evaluation_count"]=
                diagnostics.seeded_backbone_evaluation_count;
            result["polar_reverse_count"]=
                diagnostics.polar_reverse_count;
            result["reverse_executed"]=diagnostics.reverse_executed;
            result["product_seed_cleared"]=
                diagnostics.product_seed_cleared;
            return result;
        },
        py::arg("graph_generation"),py::arg("node_types"),
        py::arg("positions"),py::arg("cell"),py::arg("graph_pbc"),
        py::arg("node_graph_index"),py::arg("total_charge"),
        py::arg("total_spin"),py::arg("external_potential"),
        py::arg("include_forces")=false,
        py::arg("include_stress")=false,py::arg("include_dipole")=false,
        py::arg("polar_electrostatics")="auto",
        py::arg("pme_mesh")=std::array<std::int64_t,3>{64,64,64},
        py::arg("pme_spline_order")=6,
        "Execute the complete prepared MACE Polar scientific transaction and "
        "return only public outputs plus compact execution diagnostics.");

    cls.def_property_readonly("polar_whole_model_transaction_count",
        &Evaluator::whole_model_transaction_count);
    cls.def_property_readonly("polar_whole_model_reverse_count",
        &Evaluator::whole_model_reverse_count);
    cls.def_property_readonly("polar_whole_model_failure_count",
        &Evaluator::whole_model_failure_count);

    cls.def("polar_delta_native_program",
        [] (Evaluator& evaluator, const ByteArray& node_types,
            const ByteArray& positions, const ByteArray& cell,
            const ByteArray& graph_pbc,
            const ByteArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential,
            const bool compute_reverse,
            const bool materialize_outputs,
            const std::string& polar_electrostatics) {
            Chain& chain = evaluator.polar_deltas();
            if (total_charge.size() != 1 || total_spin.size() != 1
                || graph_pbc.size() != 3 || cell.size() != 9
                || external_potential.size() != 4)
                throw std::invalid_argument(
                    "the production polar program supports exactly one input "
                    "system; mixed or multi-system batches are unsupported.");
            typename Chain::Inputs inputs;
            inputs.solver_request = execution::long_range::
                parse_long_range_solver_request(polar_electrostatics);
            const auto& retained = evaluator.layer_features();
            if (retained.size()
                != static_cast<std::size_t>(chain.interaction_count()))
                throw std::invalid_argument(
                    "the polar native program requires one retained backbone "
                    "feature view per interaction layer.");
            inputs.node_feature_views.reserve(retained.size());
            for (const auto& layer : retained)
                inputs.node_feature_views.emplace_back(layer);

            inputs.node_types.reserve(
                static_cast<std::size_t>(node_types.size()));
            for (py::ssize_t index = 0; index < node_types.size(); ++index)
                inputs.node_types.push_back(
                    static_cast<int>(node_types.data()[index]));
            const auto row_major = [] (const ByteArray& array,
                std::vector<double>& destination) {
                destination.assign(array.data(), array.data() + array.size());
            };
            row_major(positions, inputs.positions);
            row_major(cell, inputs.cell);
            row_major(external_potential, inputs.external_potential);
            for (py::ssize_t index = 0; index < node_graph_index.size();
                ++index) {
                if (static_cast<int>(node_graph_index.data()[index]) != 0)
                    throw std::invalid_argument(
                        "the production polar program supports exactly one "
                        "input system and requires every node graph index to "
                        "be zero.");
                inputs.node_graph_index.push_back(
                    static_cast<int>(node_graph_index.data()[index]));
            }
            for (py::ssize_t index = 0; index < graph_pbc.size(); ++index)
                inputs.graph_pbc.push_back(
                    static_cast<int>(graph_pbc.data()[index]));
            inputs.total_charge = total_charge;
            inputs.total_spin = total_spin;

            const auto composed = chain.forward_composed_energy(
                inputs, /*materialize_public_state=*/false,
                materialize_outputs);
            const auto& state = composed.state;
            const int nodes = static_cast<int>(state.nodes);
            const int graphs = static_cast<int>(state.graphs);
            py::dict result;
            append_solver_diagnostics(result, chain);
            if (materialize_outputs) {
                result["energy"] = array_from_values(
                    composed.energy, graphs, 1);
                result["local_electron_energy"] = array_from_values(
                    composed.local_electron_energy, graphs, 1);
                result["electrostatic_energy_kspace"] = array_from_values(
                    composed.electrostatic_energy, graphs, 1);
                result["electrostatic_energy_correction"] = array_from_values(
                    composed.correction_energy, graphs, 1);
                result["external_field_energy"] = array_from_values(
                    composed.external_field_energy, graphs, 1);
                result["dipole"] = array_from_values(
                    composed.dipole, graphs, 3);
            }
            if (compute_reverse) {
                const auto reverse = chain.reverse_composed_energy(
                    inputs, state, /*materialize_learned_adjoints=*/false,
                    /*materialize_input_adjoints=*/materialize_outputs);
                const auto& adjoints = reverse.adjoints;
                using Precision = typename Evaluator::precision_type;
                Kokkos::View<const Precision**, Kokkos::LayoutRight>
                    flat_seed = chain.node_feature_seed();
                evaluator.set_layer_product_seed_flat(flat_seed);
                if (materialize_outputs) {
                    result["positions_adjoint"] = array_from_values(
                        adjoints.positions, nodes, 3);
                    result["external_potential_adjoint"] = array_from_values(
                        adjoints.external_potential,
                        static_cast<int>(adjoints.external_potential.size() / 4),
                        4);
                    result["total_charge_adjoint"] = array_from_values(
                        adjoints.total_charge,
                        static_cast<int>(adjoints.total_charge.size()), 1);
                    result["total_spin_adjoint"] = array_from_values(
                        adjoints.total_spin,
                        static_cast<int>(adjoints.total_spin.size()), 1);
                    result["strain_adjoint"] = array_from_values(
                        adjoints.strain,
                        static_cast<int>(adjoints.total_charge.size()), 9);
                    result["stress"] = array_from_values(reverse.stress,
                        static_cast<int>(adjoints.total_charge.size()), 9);
                }
            }
            return result;
        },
        py::arg("node_types"), py::arg("positions"), py::arg("cell"),
        py::arg("graph_pbc"), py::arg("node_graph_index"),
        py::arg("total_charge"), py::arg("total_spin"),
        py::arg("external_potential"),
        py::arg("compute_reverse") = false,
        py::arg("materialize_outputs") = true,
        py::arg("polar_electrostatics") = "auto",
        "Drive the production polar program from evaluator-owned retained "
        "feature views.  Learned intermediates and the flat backbone reverse "
        "seed remain in device memory; only public graph outputs and compact "
        "geometry adjoints are materialized.");

    cls.def("_collect_polar_public_outputs",
        [] (Evaluator& evaluator, const double volume,
            const std::uint64_t graph_generation,
            const bool include_forces, const bool include_stress) {
            const auto values = evaluator.collect_polar_public_outputs(
                volume, graph_generation, include_forces, include_stress);
            const int nodes = static_cast<int>(evaluator.polar_deltas()
                .position_adjoint_device().extent(0));
            py::dict result;
            result["energy"] = values.at(0);
            std::size_t offset = 1;
            if (include_forces) {
                result["forces"] = array_from_values(
                    std::vector<double>(values.begin() + offset,
                        values.begin() + offset + 3 * nodes), nodes, 3);
                offset += static_cast<std::size_t>(3 * nodes);
            }
            if (include_stress)
                result["stress"] = array_from_values(
                    std::vector<double>(values.begin() + offset, values.end()),
                    1, 6);
            return result;
        },
        py::arg("volume"), py::arg("graph_generation"),
        py::arg("include_forces"), py::arg("include_stress"),
        "Compose requested public energy, force and stress outputs in native "
        "storage before one compact host materialization.");

    cls.def_property_readonly("polar_public_outputs_storage_replacement_count",
        [] (const Evaluator& evaluator) {
            return evaluator.polar_public_outputs_storage_replacement_count();
        },
        "Number of device-storage replacements for packed public outputs.");
    cls.def_property_readonly("polar_particle_mesh_workspace_owned",
        &Evaluator::owns_polar_particle_mesh_workspace,
        "Whether this POLAR family owner holds its native particle-mesh workspace.");

    cls.def("polar_delta_chain_reverse",
        [] (Evaluator& evaluator, const ByteArray& node_features,
            const ByteArray& node_attrs, const ByteArray& positions,
            const ByteArray& cell, const ByteArray& graph_pbc,
            const ByteArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential,
            const ByteArray& local_electron_energy_seed,
            const ByteArray& spin_charge_density_seed,
            const std::vector<ByteArray>& potential_features_seeds,
            const std::vector<ByteArray>& fukui_sources_seeds,
            const ByteArray& table_seed_cos,
            const ByteArray& table_seed_sin,
            const std::string& potential_adjoint_route) {
            Chain& chain = evaluator.polar_deltas();
            typename Chain::Inputs inputs;
            const auto row_major = [] (const ByteArray& array,
                std::vector<double>& destination) {
                destination.assign(array.data(), array.data() + array.size());
            };
            row_major(node_features, inputs.node_features);
            row_major(node_attrs, inputs.node_attrs);
            row_major(positions, inputs.positions);
            row_major(cell, inputs.cell);
            row_major(external_potential, inputs.external_potential);
            for (py::ssize_t index = 0; index < node_graph_index.size();
                ++index)
                inputs.node_graph_index.push_back(
                    static_cast<int>(node_graph_index.data()[index]));
            for (py::ssize_t index = 0; index < graph_pbc.size(); ++index)
                inputs.graph_pbc.push_back(
                    static_cast<int>(graph_pbc.data()[index]));
            inputs.total_charge = total_charge;
            inputs.total_spin = total_spin;
            const auto state = chain.forward(inputs);
            const int nodes = static_cast<int>(state.nodes);
            typename Chain::Seeds seeds;
            if (local_electron_energy_seed.size() != 0)
                seeds.local_electron_energy.assign(
                    local_electron_energy_seed.data(),
                    local_electron_energy_seed.data()
                        + local_electron_energy_seed.size());
            if (spin_charge_density_seed.size() != 0)
                seeds.spin_charge_density.assign(
                    spin_charge_density_seed.data(),
                    spin_charge_density_seed.data()
                        + spin_charge_density_seed.size());
            for (const auto& seed : potential_features_seeds)
                seeds.potential_features.emplace_back(
                    seed.data(), seed.data() + seed.size());
            for (const auto& seed : fukui_sources_seeds)
                seeds.fukui_sources.emplace_back(
                    seed.data(), seed.data() + seed.size());
            if (table_seed_cos.size() != 0)
                seeds.table_seed_cos.assign(
                    table_seed_cos.data(),
                    table_seed_cos.data() + table_seed_cos.size());
            if (table_seed_sin.size() != 0)
                seeds.table_seed_sin.assign(
                    table_seed_sin.data(),
                    table_seed_sin.data() + table_seed_sin.size());
            if (potential_adjoint_route != "selected"
                && potential_adjoint_route != "reference")
                throw std::invalid_argument(
                    "potential_adjoint_route must be 'selected' or 'reference'.");
            const auto adjoints = chain.reverse(inputs, seeds, true, nullptr,
                nullptr, true, false,
                potential_adjoint_route == "reference");
            py::dict result;
            const int layers = chain.interaction_count();
            const int width = chain.field_update().node_feature_dimension();
            result["node_features"] = array_from_values(
                adjoints.node_features, nodes, layers * width);
            result["node_attrs"] = array_from_values(
                adjoints.node_attrs, nodes, chain.field_update()
                    .source_embedding().input_dimension());
            result["positions"] = array_from_values(
                adjoints.positions, nodes, 3);
            result["external_potential"] = array_from_values(
                adjoints.external_potential,
                static_cast<int>(adjoints.external_potential.size() / 4), 4);
            result["total_charge"] = array_from_values(adjoints.total_charge,
                static_cast<int>(adjoints.total_charge.size()), 1);
            result["total_spin"] = array_from_values(adjoints.total_spin,
                static_cast<int>(adjoints.total_spin.size()), 1);
            result["grad_cos_table"] = array_from_values(
                adjoints.grad_cos_table,
                static_cast<int>(chain.geometry_cache().cos_table.extent(0)),
                nodes);
            result["grad_sin_table"] = array_from_values(
                adjoints.grad_sin_table,
                static_cast<int>(chain.geometry_cache().sin_table.extent(0)),
                nodes);
            result["strain"] = array_from_values(adjoints.strain,
                static_cast<int>(adjoints.total_charge.size()), 9);
            return result;
        },
        py::arg("node_features"), py::arg("node_attrs"), py::arg("positions"),
        py::arg("cell"), py::arg("graph_pbc"), py::arg("node_graph_index"),
        py::arg("total_charge"), py::arg("total_spin"),
        py::arg("external_potential"),
        py::arg("local_electron_energy_seed"),
        py::arg("spin_charge_density_seed"),
        py::arg("potential_features_seeds"),
        py::arg("fukui_sources_seeds"),
        py::arg("table_seeds_cos") = ByteArray(),
        py::arg("table_seeds_sin") = ByteArray(),
        py::arg("potential_adjoint_route") = "selected",
        "Drive the composed polar chain's reverse pass.  ``table_seeds_cos`` "
        "and ``table_seeds_sin`` are the caller's own seeds of the shared "
        "cache's tables (the long-range energy path's adjoint); the chain adds "
        "them before its single geometry adjoint fires, so the position "
        "gradient composes both paths exactly once.");

    cls.def("polar_delta_composed_reverse",
        [] (Evaluator& evaluator, const ByteArray& node_features,
            const ByteArray& node_attrs, const ByteArray& positions,
            const ByteArray& cell, const ByteArray& graph_pbc,
            const ByteArray& node_graph_index,
            const std::vector<double>& total_charge,
            const std::vector<double>& total_spin,
            const ByteArray& external_potential) {
            Chain& chain = evaluator.polar_deltas();
            typename Chain::Inputs inputs;
            const auto row_major = [] (const ByteArray& array,
                std::vector<double>& destination) {
                destination.assign(array.data(), array.data() + array.size());
            };
            row_major(node_features, inputs.node_features);
            row_major(node_attrs, inputs.node_attrs);
            row_major(positions, inputs.positions);
            row_major(cell, inputs.cell);
            row_major(external_potential, inputs.external_potential);
            for (py::ssize_t index = 0; index < node_graph_index.size();
                ++index)
                inputs.node_graph_index.push_back(
                    static_cast<int>(node_graph_index.data()[index]));
            for (py::ssize_t index = 0; index < graph_pbc.size(); ++index)
                inputs.graph_pbc.push_back(
                    static_cast<int>(graph_pbc.data()[index]));
            inputs.total_charge = total_charge;
            inputs.total_spin = total_spin;

            // The reverse consumes the resident density and graph multipoles
            // produced by the complete energy forward, not merely the delta
            // chain state.  Running only ``forward`` made this legacy
            // diagnostic entry point depend on whatever composed evaluation
            // happened to precede it on the evaluator.
            const auto forward_composed = chain.forward_composed_energy(
                inputs, /*materialize_public_state=*/false,
                /*materialize_outputs=*/false);
            const auto& state = forward_composed.state;
            const auto composed =
                chain.reverse_composed_energy(inputs, state);
            const auto& adjoints = composed.adjoints;
            const int nodes = static_cast<int>(state.nodes);
            const int layers = chain.interaction_count();
            const int width = chain.field_update().node_feature_dimension();
            py::dict result;
            result["node_features"] = array_from_values(
                adjoints.node_features, nodes, layers * width);
            result["node_attrs"] = array_from_values(
                adjoints.node_attrs, nodes, chain.field_update()
                    .source_embedding().input_dimension());
            result["positions"] = array_from_values(
                adjoints.positions, nodes, 3);
            result["external_potential"] = array_from_values(
                adjoints.external_potential,
                static_cast<int>(adjoints.external_potential.size() / 4), 4);
            result["total_charge"] = array_from_values(adjoints.total_charge,
                static_cast<int>(adjoints.total_charge.size()), 1);
            result["total_spin"] = array_from_values(adjoints.total_spin,
                static_cast<int>(adjoints.total_spin.size()), 1);
            result["strain"] = array_from_values(adjoints.strain,
                static_cast<int>(adjoints.total_charge.size()), 9);
            result["stress"] = array_from_values(composed.stress,
                static_cast<int>(adjoints.total_charge.size()), 9);
            return result;
        },
        py::arg("node_features"), py::arg("node_attrs"),
        py::arg("positions"), py::arg("cell"), py::arg("graph_pbc"),
        py::arg("node_graph_index"), py::arg("total_charge"),
        py::arg("total_spin"), py::arg("external_potential"),
        "Drive the production polar energy reverse entirely in native code. "
        "The energy, correction, dipole, chain, and geometry adjoints share "
        "device-resident phase-table accumulators; only compact final "
        "adjoints are returned.");
}

} // namespace symmetrix::mace_polar

// The two family-owned instantiations the Polar bindings call.
template void
symmetrix::mace_polar::bind_polar_delta_surface<MacePolarKokkosT<double>>(
    pybind11::class_<MacePolarKokkosT<double>>& cls);
template void
symmetrix::mace_polar::bind_polar_delta_surface<MacePolarKokkosT<float>>(
    pybind11::class_<MacePolarKokkosT<float>>& cls);
