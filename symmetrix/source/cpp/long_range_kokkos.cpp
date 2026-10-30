#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <vector>

#include "long_range_kernels.hpp"

namespace py = pybind11;

using symmetrix::execution::long_range::LongRangeConstMatrix;
using symmetrix::execution::long_range::LongRangeConstTensor3;
using symmetrix::execution::long_range::LongRangeConstVector;
using symmetrix::execution::long_range::LongRangeGeometryCache;
using symmetrix::execution::long_range::LongRangeGeometryCacheFloat;
using symmetrix::execution::long_range::LongRangeGeometryMode;
using symmetrix::execution::long_range::LongRangeMatrix;
using symmetrix::execution::long_range::LongRangeVector;
using symmetrix::execution::long_range::PolarLongRangeOps;
using symmetrix::execution::long_range::PolarLongRangeOpsFloat;

namespace {

using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;

[[noreturn]] void shape_error(
    const char* what, const std::vector<py::ssize_t>& expected,
    const py::array& got)
{
    std::string message = what;
    message += " expects shape (";
    for (std::size_t i = 0; i < expected.size(); ++i) {
        if (i != 0)
            message += ", ";
        message += expected[i] < 0 ? std::string("*")
                                   : std::to_string(expected[i]);
    }
    message += "), got (";
    const py::tuple shape = got.attr("shape");
    for (py::ssize_t i = 0; i < py::len(shape); ++i) {
        if (i != 0)
            message += ", ";
        message += std::to_string(shape[i].cast<py::ssize_t>());
    }
    message += ").";
    throw std::invalid_argument(message);
}

// Zero-copy wraps of C-contiguous NumPy buffers into the kernels' unmanaged
// host views; the arrays stay alive for the duration of the call.
LongRangeConstVector<double> double_vector(
    const DoubleArray& array, const char* what)
{
    if (array.ndim() != 1)
        shape_error(what, {-1}, array);
    return LongRangeConstVector<double>(
        array.data(), static_cast<std::size_t>(array.shape(0)));
}

LongRangeVector<double> double_vector_copy(
    const DoubleArray& array, const char* what)
{
    if (array.ndim() != 1)
        shape_error(what, {-1}, array);
    LongRangeVector<double> view(
        "long_range_copy", static_cast<std::size_t>(array.shape(0)));
    std::copy(array.data(), array.data() + array.shape(0), view.data());
    return view;
}

LongRangeConstMatrix<double> double_matrix(
    const DoubleArray& array, const char* what)
{
    if (array.ndim() != 2)
        shape_error(what, {-1, -1}, array);
    return LongRangeConstMatrix<double>(array.data(),
        static_cast<std::size_t>(array.shape(0)),
        static_cast<std::size_t>(array.shape(1)));
}

LongRangeConstTensor3<double> double_tensor3(
    const DoubleArray& array, const char* what)
{
    if (array.ndim() != 3)
        shape_error(what, {-1, -1, -1}, array);
    return LongRangeConstTensor3<double>(array.data(),
        static_cast<std::size_t>(array.shape(0)),
        static_cast<std::size_t>(array.shape(1)),
        static_cast<std::size_t>(array.shape(2)));
}

LongRangeConstVector<int> int_vector(const IntArray& array, const char* what)
{
    if (array.ndim() != 1)
        shape_error(what, {-1}, array);
    return LongRangeConstVector<int>(
        array.data(), static_cast<std::size_t>(array.shape(0)));
}

LongRangeConstMatrix<int> int_matrix(const IntArray& array, const char* what)
{
    if (array.ndim() != 2)
        shape_error(what, {-1, -1}, array);
    return LongRangeConstMatrix<int>(array.data(),
        static_cast<std::size_t>(array.shape(0)),
        static_cast<std::size_t>(array.shape(1)));
}

py::array_t<double> as_array(const LongRangeVector<double>& view)
{
    py::array_t<double> array(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(view.extent(0))});
    std::copy(view.data(), view.data() + view.extent(0), array.mutable_data());
    return array;
}

py::array_t<double> as_array(const LongRangeMatrix<double>& view)
{
    py::array_t<double> array(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(view.extent(0)),
        static_cast<py::ssize_t>(view.extent(1))});
    double* out = array.mutable_data();
    for (std::size_t i = 0; i < view.extent(0); ++i)
        for (std::size_t j = 0; j < view.extent(1); ++j)
            out[i * view.extent(1) + j] = view(i, j);
    return array;
}

py::array_t<double> as_array(
    const symmetrix::execution::long_range::LongRangeTensor3<double>& view)
{
    py::array_t<double> array(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(view.extent(0)),
        static_cast<py::ssize_t>(view.extent(1)),
        static_cast<py::ssize_t>(view.extent(2))});
    double* out = array.mutable_data();
    for (std::size_t i = 0; i < view.extent(0); ++i)
        for (std::size_t j = 0; j < view.extent(1); ++j)
            for (std::size_t k = 0; k < view.extent(2); ++k)
                out[i * view.extent(1) * view.extent(2)
                    + j * view.extent(2) + k] = view(i, j, k);
    return array;
}

py::array_t<double> as_array(
    const symmetrix::execution::long_range::LongRangeTensor4<double>& view)
{
    py::array_t<double> array(std::vector<py::ssize_t>{
        static_cast<py::ssize_t>(view.extent(0)),
        static_cast<py::ssize_t>(view.extent(1)),
        static_cast<py::ssize_t>(view.extent(2)),
        static_cast<py::ssize_t>(view.extent(3))});
    double* out = array.mutable_data();
    std::size_t stride_3 = view.extent(2) * view.extent(3);
    std::size_t stride_2 = view.extent(3);
    for (std::size_t i = 0; i < view.extent(0); ++i)
        for (std::size_t j = 0; j < view.extent(1); ++j)
            for (std::size_t k = 0; k < view.extent(2); ++k)
                for (std::size_t l = 0; l < view.extent(3); ++l)
                    out[i * view.extent(1) * stride_3 + j * stride_3
                        + k * stride_2 + l] = view(i, j, k, l);
    return array;
}

py::array_t<int> as_array(const LongRangeVector<int>& view)
{
    py::array_t<int> array(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(view.extent(0))});
    std::copy(view.data(), view.data() + view.extent(0), array.mutable_data());
    return array;
}


// The cache's tables live in the kernel space (the device on a CUDA
// build): download() mirrors them into host views for the table dumps
// below.  On a host-only build the mirror aliases the view itself and no
// byte moves.
template <typename ViewType>
auto download(const ViewType& view)
    -> decltype(Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view))
{
    return Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
}

const char* geometry_mode_name(const LongRangeGeometryMode mode)
{
    switch (mode) {
    case LongRangeGeometryMode::realspace:
        return "realspace";
    case LongRangeGeometryMode::direct_reciprocal:
        return "direct_reciprocal";
    case LongRangeGeometryMode::empty:
        return "empty";
    }
    return "empty";
}

} // namespace

void bind_long_range_kokkos(py::module_& m)
{
    // The geometry cache is an opaque handle: its outputs are consumed
    // under their production names, which is the one-shared-cache contract
    // the declarative layer asserts at the binding level.
    py::class_<LongRangeGeometryCache>(m, "LongRangeGeometryCache")
        .def_property_readonly("n_k",
            [] (const LongRangeGeometryCache& cache) {
                return static_cast<py::ssize_t>(cache.n_k);
            })
        .def_property_readonly("n_nodes",
            [] (const LongRangeGeometryCache& cache) {
                return static_cast<py::ssize_t>(cache.n_nodes);
            })
        .def_property_readonly("n_graphs",
            [] (const LongRangeGeometryCache& cache) {
                return static_cast<py::ssize_t>(cache.n_graphs);
            })
        .def_property_readonly("mode",
            [] (const LongRangeGeometryCache& cache) {
                return geometry_mode_name(cache.mode);
            })
        .def_property_readonly("realspace_retained_elements",
            [] (const LongRangeGeometryCache& cache) {
                return cache.realspace_allocation_usage().elements;
            })
        .def_property_readonly("realspace_retained_bytes",
            [] (const LongRangeGeometryCache& cache) {
                return cache.realspace_allocation_usage().bytes;
            })
        .def_property_readonly("reciprocal_allocation_elements",
            [] (const LongRangeGeometryCache& cache) {
                return cache.reciprocal_allocation_usage().elements;
            })
        .def_property_readonly("reciprocal_allocation_bytes",
            [] (const LongRangeGeometryCache& cache) {
                return cache.reciprocal_allocation_usage().bytes;
            });

    py::class_<PolarLongRangeOps>(m, "PolarLongRangeOps")
        .def(py::init<double, int, double, int, std::vector<double>>(),
            py::arg("kspace_cutoff"), py::arg("density_max_l"),
            py::arg("density_smearing_width"), py::arg("feature_max_l"),
            py::arg("feature_smearing_widths"))
        .def("reciprocal_mode_count",
            [] (const PolarLongRangeOps& ops, const DoubleArray& cell,
                const std::size_t maximum_count) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                return ops.reciprocal_mode_count_up_to(
                    double_tensor3(cell, "cell"), maximum_count);
            },
            py::arg("cell"),
            py::arg("maximum_count") = std::numeric_limits<std::size_t>::max(),
            "Count the production reciprocal modes without atom-mode tables, "
            "optionally stopping at a decision threshold.")
        .def("geometry_cache",
            [] (const PolarLongRangeOps& ops, const DoubleArray& positions,
                const DoubleArray& cell, const IntArray& node_graph_index) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    shape_error("positions", {-1, 3}, positions);
                LongRangeGeometryCache cache = ops.geometry_cache(
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"));
                py::dict result;
                result["k_vectors"] = as_array(download(cache.k_vectors));
                result["k_norm2"] = as_array(download(cache.k_norm2));
                result["k_graph_index"] = as_array(download(cache.k_graph_index));
                result["k0_mask"] = as_array(download(cache.k0_mask));
                result["volume_per_k"] = as_array(download(cache.volume_per_k));
                result["k_factor_coulomb"] = as_array(download(cache.k_factor_coulomb));
                result["k_factor_proj"] = as_array(download(cache.k_factor_proj));
                result["cos_table"] = as_array(download(cache.cos_table));
                result["sin_table"] = as_array(download(cache.sin_table));
                result["density_fourier_factors"] =
                    as_array(download(cache.density_fourier_factors));
                result["feature_fourier_factors"] =
                    as_array(download(cache.feature_fourier_factors));
                // The cache handle rides along under a reserved key (last, so
                // the table copies above read the live object) -- the stage
                // methods then consume the very cache that produced these
                // tables, which is the one-shared-cache contract.
                result["cache"] = py::cast(std::move(cache));
                return result;
            },
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            "One shared cache per evaluation: k-grid, tables, per-k factors, "
            "and the analytic GTO Fourier factors.")
        .def("realspace_geometry_cache",
            [] (const PolarLongRangeOps& ops, const DoubleArray& positions,
                const DoubleArray& cell, const IntArray& node_graph_index) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    shape_error("positions", {-1, 3}, positions);
                return ops.realspace_geometry_cache(
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"));
            },
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            "Compact real-space cache with no reciprocal tables.")
        .def("realspace_geometry_cache_into",
            [] (const PolarLongRangeOps& ops, LongRangeGeometryCache& cache,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    shape_error("positions", {-1, 3}, positions);
                ops.realspace_geometry_cache_into(
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"), cache);
            },
            py::arg("cache"), py::arg("positions"), py::arg("cell"),
            py::arg("node_graph_index"),
            "Replace a same-owner cache with compact real-space geometry.")
        .def("realspace_features",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles,
                const bool include_self_interaction) {
                return as_array(ops.realspace_features(cache,
                    double_matrix(multipoles, "multipoles"),
                    include_self_interaction));
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("include_self_interaction") = false)
        .def("realspace_energy",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles,
                const bool include_self_interaction,
                const bool use_workspace) {
                return as_array(ops.realspace_energy(cache,
                    double_matrix(multipoles, "multipoles"),
                    include_self_interaction, use_workspace));
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false)
        .def("diagnostic_adjoint_realspace_features",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles,
                const DoubleArray& grad_features,
                const bool include_self_interaction,
                const bool use_workspace) {
                const auto result =
                    ops.diagnostic_adjoint_realspace_features(cache,
                        double_matrix(multipoles, "multipoles"),
                        double_matrix(grad_features, "grad_features"),
                        include_self_interaction, use_workspace);
                py::dict out;
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_cell"] = as_array(result.grad_cell);
                return out;
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("grad_features"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false,
            "Diagnostic/test-only host materialization of the device-resident "
            "real-space feature reverse stage.")
        .def("diagnostic_adjoint_realspace_energy",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles,
                const DoubleArray& grad_energy,
                const bool include_self_interaction,
                const bool use_workspace) {
                const auto result = ops.diagnostic_adjoint_realspace_energy(
                    cache, double_matrix(multipoles, "multipoles"),
                    double_vector(grad_energy, "grad_energy"),
                    include_self_interaction, use_workspace);
                py::dict out;
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_cell"] = as_array(result.grad_cell);
                return out;
            },
            py::arg("cache"), py::arg("multipoles"), py::arg("grad_energy"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false,
            "Diagnostic/test-only host materialization of the device-resident "
            "real-space energy reverse stage.")
        .def("set_realspace_reverse_strategy",
            &PolarLongRangeOps::set_realspace_reverse_strategy,
            py::arg("strategy"))
        .def("set_realspace_materialized_budget_bytes",
            &PolarLongRangeOps::set_realspace_materialized_budget_bytes,
            py::arg("budget_bytes"))
        .def("checked_realspace_atomic_accumulator_bytes",
            &PolarLongRangeOps::checked_realspace_atomic_accumulator_bytes,
            py::arg("n_nodes"))
        .def("checked_realspace_materialized_table_bytes",
            &PolarLongRangeOps::checked_realspace_materialized_table_bytes,
            py::arg("n_nodes"))
        .def("checked_realspace_full_materialized_bytes",
            &PolarLongRangeOps::checked_realspace_full_materialized_bytes,
            py::arg("n_nodes"), py::arg("include_gradients"))
        .def_property_readonly("realspace_reverse_strategy",
            [] (const PolarLongRangeOps& ops) {
                return std::string(to_string(ops.realspace_reverse_strategy()));
            })
        .def_property_readonly("realspace_reverse_strategy_requested",
            [] (const PolarLongRangeOps& ops) {
                return std::string(
                    to_string(ops.realspace_reverse_strategy_request()));
            })
        .def_property_readonly("realspace_reverse_strategy_resolved",
            [] (const PolarLongRangeOps& ops) {
                return std::string(
                    ops.realspace_reverse_strategy_resolved());
            })
        .def_property_readonly("realspace_reverse_strategy_reason",
            [] (const PolarLongRangeOps& ops) {
                return std::string(ops.realspace_reverse_strategy_reason());
            })
        .def_property_readonly("workspace_allocation_elements",
            [] (const PolarLongRangeOps& ops) {
                return ops.workspace_allocation_usage().elements;
            })
        .def_property_readonly("workspace_allocation_bytes",
            [] (const PolarLongRangeOps& ops) {
                return ops.workspace_allocation_usage().bytes;
            })
        .def("realspace_energy_launch_diagnostics",
            [] (const PolarLongRangeOps& ops) {
                const auto& diagnostics =
                    ops.realspace_energy_launch_diagnostics();
                py::dict result;
                result["selected_count"] = diagnostics.selected_count;
                result["receiver_replica_league_extent"] =
                    diagnostics.receiver_replica_league_extent;
                result["graph_reduction_league_extent"] =
                    diagnostics.graph_reduction_league_extent;
                result["partial_elements"] = diagnostics.partial_elements;
                result["full_materialized_serial_selected_count"] =
                    diagnostics.full_materialized_serial_selected_count;
                result["full_materialized_parallel_selected_count"] =
                    diagnostics.full_materialized_parallel_selected_count;
                result["full_materialized_parallel_edge_extent"] =
                    diagnostics.full_materialized_parallel_edge_extent;
                return result;
            })
        .def("realspace_reverse_launch_diagnostics",
            [] (const PolarLongRangeOps& ops) {
                const auto& diagnostics =
                    ops.realspace_reverse_launch_diagnostics();
                py::dict result;
                result["feature_selected_count"] =
                    diagnostics.feature_selected_count;
                result["feature_multipole_league_extent"] =
                    diagnostics.feature_multipole_league_extent;
                result["feature_position_league_extent"] =
                    diagnostics.feature_position_league_extent;
                result["energy_selected_count"] =
                    diagnostics.energy_selected_count;
                result["energy_multipole_league_extent"] =
                    diagnostics.energy_multipole_league_extent;
                result["energy_position_league_extent"] =
                    diagnostics.energy_position_league_extent;
                result["strategy"] =
                    std::string(to_string(ops.realspace_reverse_strategy()));
                result["requested_strategy"] = std::string(
                    to_string(ops.realspace_reverse_strategy_request()));
                result["resolved_strategy"] = std::string(
                    ops.realspace_reverse_strategy_resolved());
                result["strategy_reason"] =
                    std::string(ops.realspace_reverse_strategy_reason());
                result["feature_owner_selected_count"] =
                    diagnostics.feature_owner_selected_count;
                result["energy_owner_selected_count"] =
                    diagnostics.energy_owner_selected_count;
                result["feature_pair_atomic_selected_count"] =
                    diagnostics.feature_pair_atomic_selected_count;
                result["feature_pair_league_extent"] =
                    diagnostics.feature_pair_league_extent;
                result["energy_pair_atomic_selected_count"] =
                    diagnostics.energy_pair_atomic_selected_count;
                result["energy_pair_league_extent"] =
                    diagnostics.energy_pair_league_extent;
                result["atomic_accumulator_bytes"] =
                    diagnostics.atomic_accumulator_bytes;
                result["feature_pair_tile_selected_count"] =
                    diagnostics.feature_pair_tile_selected_count;
                result["feature_pair_tile_league_extent"] =
                    diagnostics.feature_pair_tile_league_extent;
                result["energy_pair_tile_selected_count"] =
                    diagnostics.energy_pair_tile_selected_count;
                result["energy_pair_tile_league_extent"] =
                    diagnostics.energy_pair_tile_league_extent;
                result["tile_width"] = diagnostics.tile_width;
                result["partial_accumulator_bytes"] =
                    diagnostics.partial_accumulator_bytes;
                result["feature_materialized_selected_count"] =
                    diagnostics.feature_materialized_selected_count;
                result["energy_materialized_selected_count"] =
                    diagnostics.energy_materialized_selected_count;
                result["materialized_table_build_count"] =
                    diagnostics.materialized_table_build_count;
                result["materialized_table_reuse_count"] =
                    diagnostics.materialized_table_reuse_count;
                result["materialized_table_invalidation_count"] =
                    diagnostics.materialized_table_invalidation_count;
                result["materialized_admission_rejection_count"] =
                    diagnostics.materialized_admission_rejection_count;
                result["materialized_radial_entries"] =
                    diagnostics.materialized_radial_entries;
                result["materialized_radial_bytes"] =
                    diagnostics.materialized_radial_bytes;
                result["full_materialized_table_build_count"] =
                    diagnostics.full_materialized_table_build_count;
                result["full_materialized_table_reuse_count"] =
                    diagnostics.full_materialized_table_reuse_count;
                result["full_materialized_table_invalidation_count"] =
                    diagnostics.full_materialized_table_invalidation_count;
                result["full_materialized_admission_rejection_count"] =
                    diagnostics.full_materialized_admission_rejection_count;
                result["full_materialized_feature_directed_edges"] =
                    diagnostics.full_materialized_feature_directed_edges;
                result["full_materialized_energy_directed_edges"] =
                    diagnostics.full_materialized_energy_directed_edges;
                result["full_materialized_admission_bytes"] =
                    diagnostics.full_materialized_admission_bytes;
                result["full_materialized_retained_bytes"] =
                    diagnostics.full_materialized_retained_bytes;
                result["full_materialized_has_gradients"] =
                    diagnostics.full_materialized_has_gradients;
                result["materialized_forward_reducer"] = "segmented";
                result["full_materialized_feature_serial_selected_count"] =
                    diagnostics.full_materialized_feature_serial_selected_count;
                result["full_materialized_feature_atomic_selected_count"] =
                    diagnostics.full_materialized_feature_atomic_selected_count;
                result["full_materialized_feature_segmented_selected_count"] =
                    diagnostics.full_materialized_feature_segmented_selected_count;
                result["full_materialized_feature_parallel_edge_extent"] =
                    diagnostics.full_materialized_feature_parallel_edge_extent;
                result["full_materialized_feature_pair_atomic_reverse_count"] =
                    diagnostics.full_materialized_feature_pair_atomic_reverse_count;
                result["full_materialized_energy_pair_atomic_reverse_count"] =
                    diagnostics.full_materialized_energy_pair_atomic_reverse_count;
                return result;
            })
        .def("assemble_fourier",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles) {
                return as_array(ops.assemble_fourier(
                    cache, double_matrix(multipoles, "multipoles")));
            },
            py::arg("cache"), py::arg("multipoles"))
        .def("apply_coulomb",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& density) {
                return as_array(ops.apply_coulomb(
                    cache, double_matrix(density, "density")));
            },
            py::arg("cache"), py::arg("density"))
        .def("project_to_features",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& potential) {
                return as_array(ops.project_to_features(
                    cache, double_matrix(potential, "potential")));
            },
            py::arg("cache"), py::arg("potential"))
        .def("coulomb_features",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& multipoles, const bool use_workspace) {
                return as_array(ops.coulomb_features(cache,
                    double_matrix(multipoles, "multipoles"), use_workspace));
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("use_workspace") = false)
        .def("coulomb_energy",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& charge_density,
                const std::optional<DoubleArray>& energy_self_fields) {
                std::optional<LongRangeMatrix<double>> self_fields;
                if (energy_self_fields.has_value())
                    self_fields = [&] {
                        const auto& array = *energy_self_fields;
                        LongRangeMatrix<double> view("long_range_self_fields",
                            static_cast<std::size_t>(array.shape(0)),
                            static_cast<std::size_t>(array.shape(1)));
                        auto unchecked = array.unchecked<2>();
                        for (py::ssize_t i = 0; i < array.shape(0); ++i)
                            for (py::ssize_t j = 0; j < array.shape(1); ++j)
                                view(static_cast<std::size_t>(i),
                                    static_cast<std::size_t>(j)) = unchecked(i, j);
                        return view;
                    }();
                return as_array(ops.coulomb_energy(
                    cache, double_matrix(charge_density, "charge_density"),
                    self_fields.has_value() ? &*self_fields : nullptr));
            },
            py::arg("cache"), py::arg("charge_density"),
            py::arg("energy_self_fields") = std::nullopt)
        .def("si_feature_constants",
            [] (const PolarLongRangeOps& ops) {
                return as_array(ops.si_feature_constants());
            })
        .def("si_energy_constants",
            [] (const PolarLongRangeOps& ops) {
                return as_array(ops.si_energy_constants());
            })
        .def("feature_self_terms",
            [] (const PolarLongRangeOps& ops, const DoubleArray& multipoles) {
                return as_array(ops.feature_self_terms(
                    double_matrix(multipoles, "multipoles")));
            },
            py::arg("multipoles"))
        .def("energy_self_fields",
            [] (const PolarLongRangeOps& ops, const DoubleArray& multipoles) {
                return as_array(ops.energy_self_fields(
                    double_matrix(multipoles, "multipoles")));
            },
            py::arg("multipoles"))
        .def("external_field_matrix",
            [] (const PolarLongRangeOps& ops) {
                return as_array(ops.external_field_matrix());
            })
        .def("external_field_features",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& external_potential,
                const DoubleArray& positions,
                const IntArray& node_graph_index,
                const std::optional<DoubleArray>& field_feature_norms) {
                LongRangeVector<double> norms;
                if (field_feature_norms.has_value())
                    norms = double_vector_copy(
                        *field_feature_norms, "field_feature_norms");
                return as_array(ops.external_field_features(
                    double_matrix(external_potential, "external_potential"),
                    double_matrix(positions, "positions"),
                    int_vector(node_graph_index, "node_graph_index"), norms));
            },
            py::arg("external_potential"), py::arg("positions"),
            py::arg("node_graph_index"),
            py::arg("field_feature_norms") = std::nullopt)
        .def("fukui_normalize",
            [] (const PolarLongRangeOps& ops, const DoubleArray& fukui_sources,
                const DoubleArray& charge_l0_alpha,
                const DoubleArray& charge_l0_beta,
                const DoubleArray& total_charge,
                const DoubleArray& total_spin,
                const IntArray& node_graph_index) {
                const auto result = ops.fukui_normalize(
                    double_matrix(fukui_sources, "fukui_sources"),
                    double_vector(charge_l0_alpha, "charge_l0_alpha"),
                    double_vector(charge_l0_beta, "charge_l0_beta"),
                    double_vector(total_charge, "total_charge"),
                    double_vector(total_spin, "total_spin"),
                    int_vector(node_graph_index, "node_graph_index"));
                py::dict out;
                out["charge_correction_alpha"] = as_array(result.alpha);
                out["charge_correction_beta"] = as_array(result.beta);
                return out;
            },
            py::arg("fukui_sources"), py::arg("charge_l0_alpha"),
            py::arg("charge_l0_beta"), py::arg("total_charge"),
            py::arg("total_spin"), py::arg("node_graph_index"))
        .def("dipole_readout",
            [] (const PolarLongRangeOps& ops, const DoubleArray& charge_density,
                const DoubleArray& positions,
                const IntArray& node_graph_index) {
                const auto result = ops.dipole_readout(
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions"),
                    int_vector(node_graph_index, "node_graph_index"));
                py::dict out;
                out["total_charge"] = as_array(result.total_charge);
                out["total_dipole"] = as_array(result.total_dipole);
                return out;
            },
            py::arg("charge_density"), py::arg("positions"),
            py::arg("node_graph_index"))
        .def("analytic_correction_energy",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                const auto result = ops.analytic_correction_energy(
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"));
                py::dict out;
                out["correction_energy"] = as_array(result.selected);
                out["molecule_correction"] = as_array(result.molecule);
                out["slab_correction"] = as_array(result.slab);
                return out;
            },
            py::arg("charge_density"), py::arg("positions"), py::arg("cell"),
            py::arg("node_graph_index"), py::arg("graph_pbc"))
        .def("analytic_correction_features",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density_alpha,
                const DoubleArray& charge_density_beta,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                const auto result = ops.analytic_correction_features(
                    double_matrix(charge_density_alpha, "charge_density_alpha"),
                    double_matrix(charge_density_beta, "charge_density_beta"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"));
                py::dict out;
                out["correction_fields_alpha"] = as_array(result.alpha);
                out["correction_fields_beta"] = as_array(result.beta);
                return out;
            },
            py::arg("charge_density_alpha"), py::arg("charge_density_beta"),
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            py::arg("graph_pbc"))
        // ------------------------------------------------------------------
        // Reverse-mode adjoints: one method per forward stage, seeded by
        // the output it differentiates.  The GEMM-adjoint stages return
        // their table seeds too -- the composed position gradient adds
        // them before adjoint_geometry fires (the shared-cache chain
        // rule).
        // ------------------------------------------------------------------
        .def("adjoint_geometry",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_cos_table,
                const DoubleArray& grad_sin_table) {
                return as_array(ops.adjoint_geometry(
                    cache, double_matrix(grad_cos_table, "grad_cos_table"),
                    double_matrix(grad_sin_table, "grad_sin_table")));
            },
            py::arg("cache"), py::arg("grad_cos_table"),
            py::arg("grad_sin_table"))
        .def("adjoint_assemble",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_density,
                const DoubleArray& multipoles) {
                const auto result = ops.adjoint_assemble(
                    cache, double_matrix(grad_density, "grad_density"),
                    double_matrix(multipoles, "multipoles"));
                py::dict out;
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                return out;
            },
            py::arg("cache"), py::arg("grad_density"), py::arg("multipoles"))
        .def("adjoint_coulomb",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_potential) {
                return as_array(ops.adjoint_coulomb(
                    cache,
                    double_matrix(grad_potential, "grad_potential")));
            },
            py::arg("cache"), py::arg("grad_potential"))
        .def("adjoint_project",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_features,
                const DoubleArray& potential) {
                const auto result = ops.adjoint_project(
                    cache, double_matrix(grad_features, "grad_features"),
                    double_matrix(potential, "potential"));
                py::dict out;
                out["grad_potential"] = as_array(result.grad_potential);
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                return out;
            },
            py::arg("cache"), py::arg("grad_features"), py::arg("potential"))
        .def("adjoint_coulomb_energy",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_energy,
                const DoubleArray& charge_density,
                const std::optional<DoubleArray>& energy_self_fields) {
                std::optional<LongRangeMatrix<double>> self_fields;
                if (energy_self_fields.has_value())
                    self_fields = [&] {
                        const auto& array = *energy_self_fields;
                        LongRangeMatrix<double> view(
                            "long_range_self_fields",
                            static_cast<std::size_t>(array.shape(0)),
                            static_cast<std::size_t>(array.shape(1)));
                        auto unchecked = array.unchecked<2>();
                        for (py::ssize_t i = 0; i < array.shape(0); ++i)
                            for (py::ssize_t j = 0; j < array.shape(1); ++j)
                                view(static_cast<std::size_t>(i),
                                    static_cast<std::size_t>(j)) =
                                    unchecked(i, j);
                        return view;
                    }();
                const auto result = ops.adjoint_coulomb_energy(
                    cache, double_vector(grad_energy, "grad_energy"),
                    double_matrix(charge_density, "charge_density"),
                    self_fields.has_value() ? &*self_fields : nullptr);
                py::dict out;
                out["grad_charge_density"] =
                    as_array(result.grad_charge_density);
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                return out;
            },
            py::arg("cache"), py::arg("grad_energy"),
            py::arg("charge_density"),
            py::arg("energy_self_fields") = std::nullopt)
        .def("adjoint_feature_self_terms",
            [] (const PolarLongRangeOps& ops, const DoubleArray& grad_terms) {
                return as_array(ops.adjoint_feature_self_terms(
                    double_matrix(grad_terms, "grad_terms")));
            },
            py::arg("grad_terms"))
        .def("adjoint_energy_self_fields",
            [] (const PolarLongRangeOps& ops, const DoubleArray& grad_fields) {
                return as_array(ops.adjoint_energy_self_fields(
                    double_matrix(grad_fields, "grad_fields")));
            },
            py::arg("grad_fields"))
        .def("adjoint_external_field",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& grad_features,
                const DoubleArray& external_potential,
                const DoubleArray& positions,
                const IntArray& node_graph_index,
                const std::optional<DoubleArray>& field_feature_norms) {
                LongRangeVector<double> norms;
                if (field_feature_norms.has_value())
                    norms = double_vector_copy(
                        *field_feature_norms, "field_feature_norms");
                const auto result = ops.adjoint_external_field(
                    double_matrix(grad_features, "grad_features"),
                    double_matrix(external_potential, "external_potential"),
                    double_matrix(positions, "positions"),
                    int_vector(node_graph_index, "node_graph_index"), norms);
                py::dict out;
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_external_potential"] =
                    as_array(result.grad_external_potential);
                return out;
            },
            py::arg("grad_features"), py::arg("external_potential"),
            py::arg("positions"), py::arg("node_graph_index"),
            py::arg("field_feature_norms") = std::nullopt)
        .def("adjoint_dipole_readout",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density,
                const DoubleArray& positions,
                const IntArray& node_graph_index,
                const DoubleArray& grad_total_charge,
                const DoubleArray& grad_total_dipole) {
                const auto result = ops.adjoint_dipole_readout(
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions"),
                    int_vector(node_graph_index, "node_graph_index"),
                    double_vector(grad_total_charge, "grad_total_charge"),
                    double_matrix(grad_total_dipole, "grad_total_dipole"));
                py::dict out;
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_charge_density"] =
                    as_array(result.grad_charge_density);
                return out;
            },
            py::arg("charge_density"), py::arg("positions"),
            py::arg("node_graph_index"), py::arg("grad_total_charge"),
            py::arg("grad_total_dipole"))
        // ------------------------------------------------------------------
        // The global scatter-and-broadcast adjoint (the Fukui normalization)
        // and the analytic corrections' position adjoints: the graph-level
        // aggregate seeds scatter back onto every node of the graph.
        // ------------------------------------------------------------------
        .def("adjoint_fukui_normalize",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& fukui_sources,
                const DoubleArray& charge_l0_alpha,
                const DoubleArray& charge_l0_beta,
                const DoubleArray& total_charge,
                const DoubleArray& total_spin,
                const IntArray& node_graph_index,
                const DoubleArray& grad_alpha,
                const DoubleArray& grad_beta) {
                const auto result = ops.adjoint_fukui_normalize(
                    double_matrix(fukui_sources, "fukui_sources"),
                    double_vector(charge_l0_alpha, "charge_l0_alpha"),
                    double_vector(charge_l0_beta, "charge_l0_beta"),
                    double_vector(total_charge, "total_charge"),
                    double_vector(total_spin, "total_spin"),
                    int_vector(node_graph_index, "node_graph_index"),
                    double_vector(grad_alpha, "grad_alpha"),
                    double_vector(grad_beta, "grad_beta"));
                py::dict out;
                out["grad_fukui_sources"] = as_array(result.grad_fukui_sources);
                out["grad_charge_l0_alpha"] =
                    as_array(result.grad_charge_l0_alpha);
                out["grad_charge_l0_beta"] =
                    as_array(result.grad_charge_l0_beta);
                out["grad_total_charge"] = as_array(result.grad_total_charge);
                out["grad_total_spin"] = as_array(result.grad_total_spin);
                return out;
            },
            py::arg("fukui_sources"), py::arg("charge_l0_alpha"),
            py::arg("charge_l0_beta"), py::arg("total_charge"),
            py::arg("total_spin"), py::arg("node_graph_index"),
            py::arg("grad_alpha"), py::arg("grad_beta"))
        .def("adjoint_correction_energy",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc,
                const DoubleArray& grad_energy) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                const auto result = ops.adjoint_correction_energy(
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"),
                    double_vector(grad_energy, "grad_energy"));
                py::dict out;
                out["grad_charge_density"] =
                    as_array(result.grad_charge_density);
                out["grad_positions"] = as_array(result.grad_positions);
                return out;
            },
            py::arg("charge_density"), py::arg("positions"), py::arg("cell"),
            py::arg("node_graph_index"), py::arg("graph_pbc"),
            py::arg("grad_energy"))
        .def("adjoint_correction_features",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density_alpha,
                const DoubleArray& charge_density_beta,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc,
                const DoubleArray& grad_fields_alpha,
                const DoubleArray& grad_fields_beta) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                const auto result = ops.adjoint_correction_features(
                    double_matrix(charge_density_alpha, "charge_density_alpha"),
                    double_matrix(charge_density_beta, "charge_density_beta"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"),
                    double_matrix(grad_fields_alpha, "grad_fields_alpha"),
                    double_matrix(grad_fields_beta, "grad_fields_beta"));
                py::dict out;
                out["grad_charge_density_alpha"] =
                    as_array(result.grad_charge_density_alpha);
                out["grad_charge_density_beta"] =
                    as_array(result.grad_charge_density_beta);
                out["grad_positions"] = as_array(result.grad_positions);
                return out;
            },
            py::arg("charge_density_alpha"), py::arg("charge_density_beta"),
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            py::arg("graph_pbc"), py::arg("grad_fields_alpha"),
            py::arg("grad_fields_beta"))
        // ------------------------------------------------------------------
        // Cell (strain) adjoints: the reciprocal-space virial.  One method
        // per forward stage, seeded by the output it differentiates; the
        // stages compose by adding the seeds of the cell-dependent cache
        // quantities (k vectors, per-k volumes, tables) before
        // adjoint_geometry_cell converts them to per-graph strain, and
        // coulomb_stress composes the full affine-deformation stress.
        // ------------------------------------------------------------------
        .def("adjoint_geometry_cell",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_k_vectors,
                const DoubleArray& grad_volume_per_k,
                const DoubleArray& grad_cos_table,
                const DoubleArray& grad_sin_table,
                const DoubleArray& positions) {
                return as_array(ops.adjoint_geometry_cell(
                    cache, double_matrix(grad_k_vectors, "grad_k_vectors"),
                    double_vector(grad_volume_per_k, "grad_volume_per_k"),
                    double_matrix(grad_cos_table, "grad_cos_table"),
                    double_matrix(grad_sin_table, "grad_sin_table"),
                    double_matrix(positions, "positions")));
            },
            py::arg("cache"), py::arg("grad_k_vectors"),
            py::arg("grad_volume_per_k"), py::arg("grad_cos_table"),
            py::arg("grad_sin_table"), py::arg("positions"))
        .def("adjoint_assemble_cell",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_density,
                const DoubleArray& multipoles) {
                const auto result = ops.adjoint_assemble_cell(
                    cache, double_matrix(grad_density, "grad_density"),
                    double_matrix(multipoles, "multipoles"));
                py::dict out;
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                out["grad_k_vectors"] = as_array(result.grad_k_vectors);
                out["grad_volume_per_k"] =
                    as_array(result.grad_volume_per_k);
                return out;
            },
            py::arg("cache"), py::arg("grad_density"), py::arg("multipoles"))
        .def("adjoint_coulomb_cell",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_potential,
                const DoubleArray& density) {
                const auto result = ops.adjoint_coulomb_cell(
                    cache, double_matrix(grad_potential, "grad_potential"),
                    double_matrix(density, "density"));
                py::dict out;
                out["grad_density"] = as_array(result.grad_density);
                out["grad_k_vectors"] = as_array(result.grad_k_vectors);
                return out;
            },
            py::arg("cache"), py::arg("grad_potential"), py::arg("density"))
        .def("adjoint_project_cell",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_features,
                const DoubleArray& potential) {
                const auto result = ops.adjoint_project_cell(
                    cache, double_matrix(grad_features, "grad_features"),
                    double_matrix(potential, "potential"));
                py::dict out;
                out["grad_potential"] = as_array(result.grad_potential);
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                out["grad_k_vectors"] = as_array(result.grad_k_vectors);
                return out;
            },
            py::arg("cache"), py::arg("grad_features"), py::arg("potential"))
        .def("adjoint_coulomb_energy_cell",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_energy,
                const DoubleArray& charge_density) {
                const auto result = ops.adjoint_coulomb_energy_cell(
                    cache, double_vector(grad_energy, "grad_energy"),
                    double_matrix(charge_density, "charge_density"));
                py::dict out;
                out["grad_cos_table"] = as_array(result.grad_cos_table);
                out["grad_sin_table"] = as_array(result.grad_sin_table);
                out["grad_k_vectors"] = as_array(result.grad_k_vectors);
                out["grad_volume_per_k"] =
                    as_array(result.grad_volume_per_k);
                return out;
            },
            py::arg("cache"), py::arg("grad_energy"),
            py::arg("charge_density"))
        .def("adjoint_correction_energy_cell",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc,
                const DoubleArray& grad_energy) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                return as_array(ops.adjoint_correction_energy_cell(
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"),
                    double_vector(grad_energy, "grad_energy")));
            },
            py::arg("charge_density"), py::arg("positions"), py::arg("cell"),
            py::arg("node_graph_index"), py::arg("graph_pbc"),
            py::arg("grad_energy"))
        .def("adjoint_correction_features_cell",
            [] (const PolarLongRangeOps& ops,
                const DoubleArray& charge_density_alpha,
                const DoubleArray& charge_density_beta,
                const DoubleArray& positions, const DoubleArray& cell,
                const IntArray& node_graph_index, const IntArray& graph_pbc,
                const DoubleArray& grad_fields_alpha,
                const DoubleArray& grad_fields_beta) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (graph_pbc.ndim() != 2 || graph_pbc.shape(1) != 3)
                    shape_error("graph_pbc", {-1, 3}, graph_pbc);
                return as_array(ops.adjoint_correction_features_cell(
                    double_matrix(charge_density_alpha, "charge_density_alpha"),
                    double_matrix(charge_density_beta, "charge_density_beta"),
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"),
                    int_matrix(graph_pbc, "graph_pbc"),
                    double_matrix(grad_fields_alpha, "grad_fields_alpha"),
                    double_matrix(grad_fields_beta, "grad_fields_beta")));
            },
            py::arg("charge_density_alpha"), py::arg("charge_density_beta"),
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            py::arg("graph_pbc"), py::arg("grad_fields_alpha"),
            py::arg("grad_fields_beta"))
        .def("coulomb_stress",
            [] (const PolarLongRangeOps& ops,
                const LongRangeGeometryCache& cache,
                const DoubleArray& grad_energy,
                const std::optional<DoubleArray>& grad_features,
                const DoubleArray& charge_density,
                const DoubleArray& positions) {
                LongRangeMatrix<double> features;
                if (grad_features.has_value()) {
                    const auto& array = *grad_features;
                    features = LongRangeMatrix<double>(
                        "long_range_stress_features",
                        static_cast<std::size_t>(array.shape(0)),
                        static_cast<std::size_t>(array.shape(1)));
                    auto unchecked = array.unchecked<2>();
                    for (py::ssize_t i = 0; i < array.shape(0); ++i)
                        for (py::ssize_t j = 0; j < array.shape(1); ++j)
                            features(static_cast<std::size_t>(i),
                                static_cast<std::size_t>(j)) =
                                unchecked(i, j);
                }
                return as_array(ops.coulomb_stress(
                    cache, double_vector(grad_energy, "grad_energy"),
                    grad_features.has_value() ? &features : nullptr,
                    double_matrix(charge_density, "charge_density"),
                    double_matrix(positions, "positions")));
            },
            py::arg("cache"), py::arg("grad_energy"),
            py::arg("grad_features") = std::nullopt,
            py::arg("charge_density"), py::arg("positions"))
        ;

    py::class_<LongRangeGeometryCacheFloat>(m, "LongRangeGeometryCacheFloat")
        .def_property_readonly("n_k",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return static_cast<py::ssize_t>(cache.n_k);
            })
        .def_property_readonly("n_nodes",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return static_cast<py::ssize_t>(cache.n_nodes);
            })
        .def_property_readonly("n_graphs",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return static_cast<py::ssize_t>(cache.n_graphs);
            })
        .def_property_readonly("mode",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return geometry_mode_name(cache.mode);
            })
        .def_property_readonly("realspace_retained_elements",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return cache.realspace_allocation_usage().elements;
            })
        .def_property_readonly("realspace_retained_bytes",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return cache.realspace_allocation_usage().bytes;
            })
        .def_property_readonly("reciprocal_allocation_elements",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return cache.reciprocal_allocation_usage().elements;
            })
        .def_property_readonly("reciprocal_allocation_bytes",
            [] (const LongRangeGeometryCacheFloat& cache) {
                return cache.reciprocal_allocation_usage().bytes;
            });

    py::class_<PolarLongRangeOpsFloat>(m, "PolarLongRangeOpsFloat")
        .def(py::init<double, int, double, int, std::vector<double>>(),
            py::arg("kspace_cutoff"), py::arg("density_max_l"),
            py::arg("density_smearing_width"), py::arg("feature_max_l"),
            py::arg("feature_smearing_widths"))
        .def("realspace_geometry_cache",
            [] (const PolarLongRangeOpsFloat& ops, const DoubleArray& positions,
                const DoubleArray& cell, const IntArray& node_graph_index) {
                if (cell.ndim() != 3 || cell.shape(1) != 3 || cell.shape(2) != 3)
                    shape_error("cell", {-1, 3, 3}, cell);
                if (positions.ndim() != 2 || positions.shape(1) != 3)
                    shape_error("positions", {-1, 3}, positions);
                return ops.realspace_geometry_cache(
                    double_matrix(positions, "positions"),
                    double_tensor3(cell, "cell"),
                    int_vector(node_graph_index, "node_graph_index"));
            },
            py::arg("positions"), py::arg("cell"), py::arg("node_graph_index"),
            "Compact fp32 real-space cache with no reciprocal tables.")
        .def("realspace_features",
            [] (const PolarLongRangeOpsFloat& ops,
                const LongRangeGeometryCacheFloat& cache,
                const DoubleArray& multipoles,
                const bool include_self_interaction) {
                return as_array(ops.realspace_features(cache,
                    double_matrix(multipoles, "multipoles"),
                    include_self_interaction));
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("include_self_interaction") = false)
        .def("realspace_energy",
            [] (const PolarLongRangeOpsFloat& ops,
                const LongRangeGeometryCacheFloat& cache,
                const DoubleArray& multipoles,
                const bool include_self_interaction,
                const bool use_workspace) {
                return as_array(ops.realspace_energy(cache,
                    double_matrix(multipoles, "multipoles"),
                    include_self_interaction, use_workspace));
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false)
        .def("diagnostic_adjoint_realspace_features",
            [] (const PolarLongRangeOpsFloat& ops,
                const LongRangeGeometryCacheFloat& cache,
                const DoubleArray& multipoles,
                const DoubleArray& grad_features,
                const bool include_self_interaction,
                const bool use_workspace) {
                const auto result =
                    ops.diagnostic_adjoint_realspace_features(cache,
                        double_matrix(multipoles, "multipoles"),
                        double_matrix(grad_features, "grad_features"),
                        include_self_interaction, use_workspace);
                py::dict out;
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_cell"] = as_array(result.grad_cell);
                return out;
            },
            py::arg("cache"), py::arg("multipoles"),
            py::arg("grad_features"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false,
            "Diagnostic/test-only host materialization of the device-resident "
            "fp32 real-space feature reverse stage.")
        .def("diagnostic_adjoint_realspace_energy",
            [] (const PolarLongRangeOpsFloat& ops,
                const LongRangeGeometryCacheFloat& cache,
                const DoubleArray& multipoles,
                const DoubleArray& grad_energy,
                const bool include_self_interaction,
                const bool use_workspace) {
                const auto result = ops.diagnostic_adjoint_realspace_energy(
                    cache, double_matrix(multipoles, "multipoles"),
                    double_vector(grad_energy, "grad_energy"),
                    include_self_interaction, use_workspace);
                py::dict out;
                out["grad_multipoles"] = as_array(result.grad_multipoles);
                out["grad_positions"] = as_array(result.grad_positions);
                out["grad_cell"] = as_array(result.grad_cell);
                return out;
            },
            py::arg("cache"), py::arg("multipoles"), py::arg("grad_energy"),
            py::arg("include_self_interaction") = false,
            py::arg("use_workspace") = false,
            "Diagnostic/test-only host materialization of the device-resident "
            "fp32 real-space energy reverse stage.")
        .def("set_realspace_reverse_strategy",
            &PolarLongRangeOpsFloat::set_realspace_reverse_strategy,
            py::arg("strategy"))
        .def("set_realspace_materialized_budget_bytes",
            &PolarLongRangeOpsFloat::set_realspace_materialized_budget_bytes,
            py::arg("budget_bytes"))
        .def("checked_realspace_atomic_accumulator_bytes",
            &PolarLongRangeOpsFloat::checked_realspace_atomic_accumulator_bytes,
            py::arg("n_nodes"))
        .def("checked_realspace_materialized_table_bytes",
            &PolarLongRangeOpsFloat::checked_realspace_materialized_table_bytes,
            py::arg("n_nodes"))
        .def("checked_realspace_full_materialized_bytes",
            &PolarLongRangeOpsFloat::checked_realspace_full_materialized_bytes,
            py::arg("n_nodes"), py::arg("include_gradients"))
        .def_property_readonly("realspace_reverse_strategy",
            [] (const PolarLongRangeOpsFloat& ops) {
                return std::string(to_string(ops.realspace_reverse_strategy()));
            })
        .def_property_readonly("realspace_reverse_strategy_requested",
            [] (const PolarLongRangeOpsFloat& ops) {
                return std::string(
                    to_string(ops.realspace_reverse_strategy_request()));
            })
        .def_property_readonly("realspace_reverse_strategy_resolved",
            [] (const PolarLongRangeOpsFloat& ops) {
                return std::string(
                    ops.realspace_reverse_strategy_resolved());
            })
        .def_property_readonly("realspace_reverse_strategy_reason",
            [] (const PolarLongRangeOpsFloat& ops) {
                return std::string(ops.realspace_reverse_strategy_reason());
            })
        .def_property_readonly("workspace_allocation_elements",
            [] (const PolarLongRangeOpsFloat& ops) {
                return ops.workspace_allocation_usage().elements;
            })
        .def_property_readonly("workspace_allocation_bytes",
            [] (const PolarLongRangeOpsFloat& ops) {
                return ops.workspace_allocation_usage().bytes;
            })
        .def("realspace_energy_launch_diagnostics",
            [] (const PolarLongRangeOpsFloat& ops) {
                const auto& diagnostics =
                    ops.realspace_energy_launch_diagnostics();
                py::dict result;
                result["selected_count"] = diagnostics.selected_count;
                result["receiver_replica_league_extent"] =
                    diagnostics.receiver_replica_league_extent;
                result["graph_reduction_league_extent"] =
                    diagnostics.graph_reduction_league_extent;
                result["partial_elements"] = diagnostics.partial_elements;
                result["full_materialized_serial_selected_count"] =
                    diagnostics.full_materialized_serial_selected_count;
                result["full_materialized_parallel_selected_count"] =
                    diagnostics.full_materialized_parallel_selected_count;
                result["full_materialized_parallel_edge_extent"] =
                    diagnostics.full_materialized_parallel_edge_extent;
                return result;
            })
        .def("realspace_reverse_launch_diagnostics",
            [] (const PolarLongRangeOpsFloat& ops) {
                const auto& diagnostics =
                    ops.realspace_reverse_launch_diagnostics();
                py::dict result;
                result["feature_selected_count"] =
                    diagnostics.feature_selected_count;
                result["feature_multipole_league_extent"] =
                    diagnostics.feature_multipole_league_extent;
                result["feature_position_league_extent"] =
                    diagnostics.feature_position_league_extent;
                result["energy_selected_count"] =
                    diagnostics.energy_selected_count;
                result["energy_multipole_league_extent"] =
                    diagnostics.energy_multipole_league_extent;
                result["energy_position_league_extent"] =
                    diagnostics.energy_position_league_extent;
                result["strategy"] =
                    std::string(to_string(ops.realspace_reverse_strategy()));
                result["requested_strategy"] = std::string(
                    to_string(ops.realspace_reverse_strategy_request()));
                result["resolved_strategy"] = std::string(
                    ops.realspace_reverse_strategy_resolved());
                result["strategy_reason"] =
                    std::string(ops.realspace_reverse_strategy_reason());
                result["feature_owner_selected_count"] =
                    diagnostics.feature_owner_selected_count;
                result["energy_owner_selected_count"] =
                    diagnostics.energy_owner_selected_count;
                result["feature_pair_atomic_selected_count"] =
                    diagnostics.feature_pair_atomic_selected_count;
                result["feature_pair_league_extent"] =
                    diagnostics.feature_pair_league_extent;
                result["energy_pair_atomic_selected_count"] =
                    diagnostics.energy_pair_atomic_selected_count;
                result["energy_pair_league_extent"] =
                    diagnostics.energy_pair_league_extent;
                result["atomic_accumulator_bytes"] =
                    diagnostics.atomic_accumulator_bytes;
                result["feature_pair_tile_selected_count"] =
                    diagnostics.feature_pair_tile_selected_count;
                result["feature_pair_tile_league_extent"] =
                    diagnostics.feature_pair_tile_league_extent;
                result["energy_pair_tile_selected_count"] =
                    diagnostics.energy_pair_tile_selected_count;
                result["energy_pair_tile_league_extent"] =
                    diagnostics.energy_pair_tile_league_extent;
                result["tile_width"] = diagnostics.tile_width;
                result["partial_accumulator_bytes"] =
                    diagnostics.partial_accumulator_bytes;
                result["feature_materialized_selected_count"] =
                    diagnostics.feature_materialized_selected_count;
                result["energy_materialized_selected_count"] =
                    diagnostics.energy_materialized_selected_count;
                result["materialized_table_build_count"] =
                    diagnostics.materialized_table_build_count;
                result["materialized_table_reuse_count"] =
                    diagnostics.materialized_table_reuse_count;
                result["materialized_table_invalidation_count"] =
                    diagnostics.materialized_table_invalidation_count;
                result["materialized_admission_rejection_count"] =
                    diagnostics.materialized_admission_rejection_count;
                result["materialized_radial_entries"] =
                    diagnostics.materialized_radial_entries;
                result["materialized_radial_bytes"] =
                    diagnostics.materialized_radial_bytes;
                result["full_materialized_table_build_count"] =
                    diagnostics.full_materialized_table_build_count;
                result["full_materialized_table_reuse_count"] =
                    diagnostics.full_materialized_table_reuse_count;
                result["full_materialized_table_invalidation_count"] =
                    diagnostics.full_materialized_table_invalidation_count;
                result["full_materialized_admission_rejection_count"] =
                    diagnostics.full_materialized_admission_rejection_count;
                result["full_materialized_feature_directed_edges"] =
                    diagnostics.full_materialized_feature_directed_edges;
                result["full_materialized_energy_directed_edges"] =
                    diagnostics.full_materialized_energy_directed_edges;
                result["full_materialized_admission_bytes"] =
                    diagnostics.full_materialized_admission_bytes;
                result["full_materialized_retained_bytes"] =
                    diagnostics.full_materialized_retained_bytes;
                result["full_materialized_has_gradients"] =
                    diagnostics.full_materialized_has_gradients;
                result["materialized_forward_reducer"] = "segmented";
                result["full_materialized_feature_serial_selected_count"] =
                    diagnostics.full_materialized_feature_serial_selected_count;
                result["full_materialized_feature_atomic_selected_count"] =
                    diagnostics.full_materialized_feature_atomic_selected_count;
                result["full_materialized_feature_segmented_selected_count"] =
                    diagnostics.full_materialized_feature_segmented_selected_count;
                result["full_materialized_feature_parallel_edge_extent"] =
                    diagnostics.full_materialized_feature_parallel_edge_extent;
                result["full_materialized_feature_pair_atomic_reverse_count"] =
                    diagnostics.full_materialized_feature_pair_atomic_reverse_count;
                result["full_materialized_energy_pair_atomic_reverse_count"] =
                    diagnostics.full_materialized_energy_pair_atomic_reverse_count;
                return result;
            });
}
