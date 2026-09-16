#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "device_environment.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <numeric>
#include <stdexcept>

#include "utilities_kokkos.hpp"
#include "mace_kokkos.hpp"

namespace py = pybind11;
using ContiguousIntArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using ContiguousDoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename T>
py::array_t<T> shaped_array(
    const std::vector<T>& values,
    const std::initializer_list<py::ssize_t> shape)
{
    std::size_t expected_size = 1;
    for (const py::ssize_t extent : shape) {
        if (extent < 0)
            throw std::invalid_argument("A NumPy export extent is negative.");
        expected_size *= static_cast<std::size_t>(extent);
    }
    if (expected_size != values.size())
        throw std::logic_error("A NumPy export shape is inconsistent.");
    py::array_t<T> result(shape);
    std::copy(values.begin(), values.end(), result.mutable_data());
    return result;
}

template <typename ViewType>
auto view_prefix_1d(const ViewType& view, const std::size_t extent)
{
    using T = typename ViewType::non_const_value_type;
    if (view.extent(0) < extent)
        throw std::logic_error("A benchmark export view is too small.");
    const auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(), view);
    auto result = std::vector<T>(extent);
    for (std::size_t i=0; i<extent; ++i)
        result[i] = host(i);
    return result;
}

template <typename ViewType>
auto view_prefix_3d(
    const ViewType& view,
    const std::size_t extent_0,
    const std::size_t extent_1,
    const std::size_t extent_2)
{
    using T = typename ViewType::non_const_value_type;
    if (view.extent(0) < extent_0
        || view.extent(1) != extent_1
        || view.extent(2) != extent_2)
        throw std::logic_error("A benchmark export tensor extent is invalid.");
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), view);
    auto result = std::vector<T>(extent_0*extent_1*extent_2);
    for (std::size_t i=0; i<extent_0; ++i)
        for (std::size_t j=0; j<extent_1; ++j)
            for (std::size_t k=0; k<extent_2; ++k)
                result[(i*extent_1+j)*extent_2+k] = host(i,j,k);
    return result;
}

template <typename Precision>
py::dict dual_layer_source_schedule_for_testing(
    const MACEKokkos<Precision>& self)
{
    if (!self.dual_layer_tiled_plan_active)
        throw std::logic_error(
            "Dual-layer source schedule is unavailable outside the tiled plan.");
    const std::size_t edges = static_cast<std::size_t>(
        self.execution_active_edges);
    const std::size_t segments = static_cast<std::size_t>(
        self.dual_layer_source_segments);
    const std::size_t tiles = self.dual_layer_tile_segment_offsets.extent(0);
    const std::size_t receivers = static_cast<std::size_t>(
        self.execution_active_receivers);
    py::dict result;
    result["tile_segment_offsets"] = shaped_array(
        view_prefix_1d(self.dual_layer_tile_segment_offsets, tiles),
        {static_cast<py::ssize_t>(tiles)});
    result["segment_source_ids"] = shaped_array(
        view_prefix_1d(self.dual_layer_segment_source_ids, segments),
        {static_cast<py::ssize_t>(segments)});
    result["segment_edge_offsets"] = shaped_array(
        view_prefix_1d(self.dual_layer_segment_edge_offsets, segments+1),
        {static_cast<py::ssize_t>(segments+1)});
    result["source_edges"] = shaped_array(
        view_prefix_1d(self.dual_layer_source_edges, edges),
        {static_cast<py::ssize_t>(edges)});
    result["num_neigh"] = shaped_array(
        view_prefix_1d(self.execution_prepared_num_neigh, receivers),
        {static_cast<py::ssize_t>(receivers)});
    result["edge_sources"] = shaped_array(
        view_prefix_1d(self.execution_prepared_neigh_indices, edges),
        {static_cast<py::ssize_t>(edges)});
    return result;
}

template <typename Precision>
py::dict factorized_operator_benchmark_outputs(
    MACEKokkos<Precision>& self,
    const std::uint64_t token)
{
    self.synchronize_factorized_operator_benchmark(token);
    const int num_nodes = self.factorized_operator_benchmark_num_nodes;
    const int num_edges = self.factorized_operator_benchmark_num_edges;
    py::dict result;
    result["expected_out"] = shaped_array(
        view_prefix_3d(
            self.A1, num_nodes, self.num_lm, self.num_channels),
        {num_nodes, self.num_lm, self.num_channels});
    result["expected_grad_x"] = shaped_array(
        view_prefix_3d(
            self.H1_adj, num_nodes, self.num_LM, self.num_channels),
        {num_nodes, self.num_LM, self.num_channels});
    result["expected_directed_force"] = shaped_array(
        view_prefix_1d(self.node_forces, 3*static_cast<std::size_t>(num_edges)),
        {num_edges, 3});
    return result;
}

template <typename Precision>
py::dict factorized_operator_export(
    MACEKokkos<Precision>& self,
    const std::uint64_t token)
{
    self.synchronize_factorized_operator_benchmark(token);
    const int num_nodes = self.factorized_operator_benchmark_num_nodes;
    const int num_edges = self.factorized_operator_benchmark_num_edges;
    const int channels = self.num_channels;
    const int embedding = self.factorized_embedding_width;

    const auto edge_sources = view_prefix_1d(
        self.execution_prepared_neigh_indices, num_edges);
    const auto edge_receivers = view_prefix_1d(
        self.execution_edge_receivers, num_edges);
    auto edge_index =
        std::vector<std::int64_t>(2*static_cast<std::size_t>(num_edges));
    std::copy(edge_sources.begin(), edge_sources.end(), edge_index.begin());
    std::copy(
        edge_receivers.begin(), edge_receivers.end(),
        edge_index.begin()+num_edges);

    py::dict result;
    result["x"] = shaped_array(
        view_prefix_3d(
            self.H1, num_nodes, self.num_LM, channels),
        {num_nodes, self.num_LM, channels});
    result["edge_index"] = shaped_array(
        edge_index, {2, num_edges});
    result["sh"] = shaped_array(
        view_prefix_1d(
            self.Y, static_cast<std::size_t>(num_edges)*self.num_lm),
        {num_edges, self.num_lm});
    result["phi"] = shaped_array(
        view2vector(self.factorized_operator_benchmark_phi),
        {num_edges, embedding});
    result["radial_linear"] = shaped_array(
        self.factorized_operator_benchmark_radial_linear(token),
        {embedding, 10*channels*channels});
    result["grad_out"] = shaped_array(
        view_prefix_3d(
            self.factorized_operator_benchmark_a1_adjoint,
            num_nodes, self.num_lm, channels),
        {num_nodes, self.num_lm, channels});
    result["xyz"] = shaped_array(
        view_prefix_1d(
            self.factorized_operator_benchmark_xyz,
            3*static_cast<std::size_t>(num_edges)),
        {num_edges, 3});
    result["r"] = shaped_array(
        view_prefix_1d(self.factorized_operator_benchmark_r, num_edges),
        {num_edges});
    result["dsh_dxyz"] = shaped_array(
        view_prefix_1d(
            self.Y_grad,
            3*static_cast<std::size_t>(num_edges)*self.num_lm),
        {num_edges, 3, self.num_lm});
    result["dphi_dr"] = shaped_array(
        view2vector(self.factorized_operator_benchmark_dphi_dr),
        {num_edges, embedding});

    const auto outputs = factorized_operator_benchmark_outputs(self, token);
    result["expected_out"] = outputs["expected_out"];
    result["expected_grad_x"] = outputs["expected_grad_x"];
    result["expected_directed_force"] =
        outputs["expected_directed_force"];
    return result;
}

template <typename Precision>
std::unique_ptr<MACEKokkos<Precision>> load_mace_kokkos(
    const std::string& filename, const std::string& requested_head)
{
    auto evaluator = std::make_unique<MACEKokkos<Precision>>(
        filename, requested_head);
    if (!evaluator->supports_streamed_edges()) {
        if (PyErr_WarnEx(
                PyExc_UserWarning,
                "Loaded legacy Symmetrix format-v1 pair-spline model; using "
                "streamed_edges='materialized'. Re-export with radial_format='compact' "
                "to enable streamed_edges='generic' execution.",
                1) < 0)
            throw py::error_already_set();
    }
    return evaluator;
}

template <typename Precision>
void prepare_active_types(
    MACEKokkos<Precision>& self,
    const ContiguousIntArray& node_types)
{
    self.prepare_active_types(std::vector<int>(
        node_types.data(), node_types.data()+node_types.size()));
}

template <typename Precision>
void prepare_active_types(
    MACEKokkos<Precision>& self,
    const ContiguousIntArray& node_types,
    const ContiguousIntArray& neigh_types)
{
    auto types = std::vector<int>();
    types.reserve(self.atomic_numbers.size());
    auto seen = std::vector<unsigned char>(self.atomic_numbers.size(), 0);
    const auto append = [&] (const ContiguousIntArray& input) {
        for (py::ssize_t index=0; index<input.size(); ++index) {
            const int type = input.data()[index];
            if (type < 0 || type >= static_cast<int>(seen.size())) {
                types.push_back(type);
            } else if (!seen[type]) {
                seen[type] = 1;
                types.push_back(type);
            }
        }
    };
    append(node_types);
    append(neigh_types);
    self.prepare_active_types(std::move(types));
}

template <typename Precision>
void bind_mace_kokkos(py::module_ &m, const char* class_name)
{
    using ContiguousPrecisionArray =
        py::array_t<Precision, py::array::c_style | py::array::forcecast>;

    py::class_<MACEKokkos<Precision>>(m, class_name)
        .def(py::init(&load_mace_kokkos<Precision>),
            py::arg("filename"), py::arg("head") = "")
        .def("set_streamed_edges", &MACEKokkos<Precision>::set_streamed_edges)
        .def("_set_execution_plan_request",
            &MACEKokkos<Precision>::set_execution_plan_request,
            py::arg("algorithm"), py::arg("profile"),
            py::arg("debug_plan") = "")
        .def("_resolve_execution_plan", &MACEKokkos<Precision>::resolve_execution_plan)
        .def_property_readonly("execution_plan_report",
            [] (const MACEKokkos<Precision>& self) {
                const auto& report = self.execution_plan_report();
                py::dict result;
                result["requested_algorithm"] =
                    mace_streamed_edges_mode_name(report.requested_algorithm);
                result["requested_profile"] =
                    symmetrix::execution::execution_profile_name(
                        report.requested_profile);
                result["selection_source"] = report.selection_source;
                result["state"] = report.state;
                result["selected_id"] = report.selected_id;
                result["selection_reason"] = report.selection_reason;
                result["available_bytes"] = report.available_bytes;
                result["reserve_bytes"] = report.reserve_bytes;
                result["boundary_attempt"] = report.boundary_attempt;
                py::list candidates;
                for (const auto& candidate : report.candidates) {
                    py::dict entry;
                    entry["id"] = candidate.id;
                    entry["qualified"] = candidate.qualified;
                    entry["estimated_bytes"] = candidate.estimated_bytes;
                    entry["reason"] = candidate.reason;
                    candidates.append(entry);
                }
                result["candidates"] = candidates;
                return result;
            })
        .def("_load_jit_cuda_plugin",
            &MACEKokkos<Precision>::load_jit_cuda_plugin,
            py::arg("path"),
            py::arg("persistent_blocks_per_compute_unit") = 4)
        .def("_load_jit_hip_plugin",
            &MACEKokkos<Precision>::load_jit_hip_plugin)
        .def("_load_jit_device_plugin",
            &MACEKokkos<Precision>::load_jit_device_plugin,
            py::arg("path"),
            py::arg("persistent_blocks_per_compute_unit") = 8)
        .def("_load_m0_device_module",
            &MACEKokkos<Precision>::load_m0_device_module,
            py::arg("path"),
            py::arg("schedule") = "chunk32",
            py::arg("persistent_blocks_per_compute_unit") = 8)
        .def("_load_m0_host_plugin",
            &MACEKokkos<Precision>::load_m0_host_plugin)
        .def("_load_r0_device_module",
            &MACEKokkos<Precision>::load_r0_device_module,
            py::arg("path"),
            py::arg("persistent_blocks_per_compute_unit") = 8)
        .def("_load_jit_host_plugin",
            &MACEKokkos<Precision>::load_jit_host_plugin)
        .def("_set_factorized_direct_forward_executor",
            &MACEKokkos<Precision>::set_factorized_direct_forward_executor)
        .def("_set_factorized_direct_reverse_executor",
            &MACEKokkos<Precision>::set_factorized_direct_reverse_executor)
        .def("_set_standard_r0_executor",
            &MACEKokkos<Precision>::set_standard_r0_executor)
        .def("_set_standard_m0_executor",
            &MACEKokkos<Precision>::set_standard_m0_executor)
        .def("_set_kernel_launch_policy",
            &MACEKokkos<Precision>::set_kernel_launch_policy)
        .def("_set_kernel_launch_profile_override",
            &MACEKokkos<Precision>::set_kernel_launch_profile_override)
        .def("_set_jit_device_plugin_launch_override",
            &MACEKokkos<Precision>::set_jit_device_plugin_launch_override)
        .def("_clear_kernel_launch_profile_overrides",
            &MACEKokkos<Precision>::clear_kernel_launch_profile_overrides)
        .def_property_readonly("kernel_launch_policy",
            &MACEKokkos<Precision>::kernel_launch_policy_name)
        .def_property_readonly("kernel_launch_profile_diagnostics",
            [] (const MACEKokkos<Precision>& self) {
                py::dict result;
                for (const auto& diagnostic :
                     self.kernel_launch_profile_diagnostics()) {
                    py::dict stage;
                    stage["implementation_id"] = diagnostic.implementation_id;
                    stage["implementation_kind"] =
                        diagnostic.implementation_kind;
                    stage["profile_id"] = diagnostic.profile_id;
                    stage["selection_source"] = diagnostic.selection_source;
                    stage["rejection_reason"] = diagnostic.rejection_reason;
                    stage["default_blocks_per_compute_unit"] =
                        diagnostic.default_blocks_per_compute_unit;
                    stage["active_blocks_per_compute_unit"] =
                        diagnostic.active_blocks_per_compute_unit;
                    stage["persistent_blocks"] = diagnostic.persistent_blocks;
                    stage["calibration_candidates"] =
                        diagnostic.calibration_candidates;
                    stage["calibration_permitted"] =
                        diagnostic.calibration_permitted;
                    result[py::str(diagnostic.stage)] = stage;
                }
                return result;
            })
        .def("_set_m1_polynomial_policy",
            &MACEKokkos<Precision>::set_m1_polynomial_policy)
        .def("_set_m1_recompute_tile_channels",
            &MACEKokkos<Precision>::set_m1_recompute_tile_channels)
        .def("_set_m1_recompute_scratch_limit_for_testing",
            &MACEKokkos<Precision>::set_m1_recompute_scratch_limit_for_testing)
        .def("_set_mh0_state_policy",
            &MACEKokkos<Precision>::set_mh0_state_policy)
        .def("_set_edge_geometry_policy",
            &MACEKokkos<Precision>::set_edge_geometry_policy)
        .def("_set_harmonic_storage_policy",
            &MACEKokkos<Precision>::set_harmonic_storage_policy)
        .def("_set_low_memory_device_memory_info_for_testing",
            &MACEKokkos<Precision>::set_low_memory_device_memory_info_for_testing)
        .def("_set_single_layer_workspace_receiver_limit_for_testing",
            &MACEKokkos<Precision>::
                set_single_layer_workspace_receiver_limit_for_testing)
        .def("_set_dual_layer_workspace_receiver_limit_for_testing",
            &MACEKokkos<Precision>::
                set_dual_layer_workspace_receiver_limit_for_testing)
        .def("_dual_layer_source_schedule_for_testing",
            &dual_layer_source_schedule_for_testing<Precision>)
        .def("_set_allow_fixed_workspace",
            &MACEKokkos<Precision>::set_allow_fixed_workspace)
        .def_property_readonly("allow_fixed_workspace",
            &MACEKokkos<Precision>::allow_fixed_workspace)
        .def("_set_low_memory",
            &MACEKokkos<Precision>::set_low_memory)
        .def("_set_readout_policy",
            &MACEKokkos<Precision>::set_readout_policy)
        .def("_set_phi1_policy",
            &MACEKokkos<Precision>::set_phi1_policy)
        .def_property_readonly("m1_polynomial_policy",
            &MACEKokkos<Precision>::m1_polynomial_policy_name)
        .def_property_readonly("m1_polynomial_policy_request",
            &MACEKokkos<Precision>::m1_polynomial_policy_request_name)
        .def_property_readonly("m1_recompute_backend",
            &MACEKokkos<Precision>::m1_recompute_backend_name)
        .def_property_readonly("m1_recompute_fallback_reason",
            &MACEKokkos<Precision>::m1_recompute_fallback_reason_name)
        .def_property_readonly("m1_recompute_tile_channels",
            [] (const MACEKokkos<Precision>& self) {
                return self.m1_recompute_tile_channels;
            })
        .def_property_readonly("m1_poly_values_active_bytes",
            &MACEKokkos<Precision>::m1_poly_values_active_bytes)
        .def_property_readonly("m1_poly_values_capacity_bytes",
            &MACEKokkos<Precision>::m1_poly_values_capacity_bytes)
        .def_property_readonly("m1_poly_adjoints_active_bytes",
            &MACEKokkos<Precision>::m1_poly_adjoints_active_bytes)
        .def_property_readonly("m1_poly_adjoints_capacity_bytes",
            &MACEKokkos<Precision>::m1_poly_adjoints_capacity_bytes)
        .def_property_readonly("m1_recompute_scratch_bytes",
            &MACEKokkos<Precision>::m1_recompute_scratch_bytes)
        .def_property_readonly("m1_recompute_scratch_limit_bytes",
            &MACEKokkos<Precision>::m1_recompute_scratch_limit_bytes)
        .def_property_readonly("standard_m1_module_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m1_module_ready;
            })
        .def_property_readonly("standard_m1_module_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m1_module_forward_launch_count;
            })
        .def_property_readonly("standard_m1_module_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m1_module_reverse_launch_count;
            })
        .def_property_readonly("mh0_state_policy",
            &MACEKokkos<Precision>::mh0_state_policy_name)
        .def_property_readonly("mh0_state_policy_request",
            &MACEKokkos<Precision>::mh0_state_policy_request_name)
        .def_property_readonly("mh0_state_policy_fallback_reason",
            &MACEKokkos<Precision>::mh0_state_policy_fallback_reason_name)
        .def_property_readonly("mh0_reused_state_bytes",
            &MACEKokkos<Precision>::mh0_reused_state_bytes)
        .def_property_readonly("mh0_auxiliary_state_bytes",
            &MACEKokkos<Precision>::mh0_auxiliary_state_bytes)
        .def_property_readonly("h1_m0_adjoint_ping_pong_active",
            &MACEKokkos<Precision>::h1_m0_adjoint_ping_pong_active)
        .def_property_readonly("h1_m0_adjoint_ping_pong_bytes",
            &MACEKokkos<Precision>::h1_m0_adjoint_ping_pong_bytes)
        .def_property_readonly("mh0_m0_adjoint_alias_active",
            &MACEKokkos<Precision>::mh0_m0_adjoint_alias_active)
        .def_property_readonly("mh0_m0_forward_replacement_count",
            &MACEKokkos<Precision>::mh0_m0_forward_replacement_count_value)
        .def_property_readonly("mh0_m0_adjoint_allocation_count",
            &MACEKokkos<Precision>::mh0_m0_adjoint_allocation_count_value)
        .def_property_readonly("mh0_m0_alias_detach_count",
            &MACEKokkos<Precision>::mh0_m0_alias_detach_count_value)
        .def_property_readonly("readout_workspace_bytes",
            &MACEKokkos<Precision>::readout_workspace_bytes)
        .def_property_readonly("readout_policy",
            &MACEKokkos<Precision>::readout_policy_name)
        .def_property_readonly("phi1_policy",
            &MACEKokkos<Precision>::phi1_policy_name)
        .def_property_readonly("phi1_workspace_bytes",
            &MACEKokkos<Precision>::phi1_workspace_bytes)
        .def_property_readonly("generic_phi1_source_adjoint_workspace_bytes",
            &MACEKokkos<Precision>::generic_phi1_source_adjoint_workspace_bytes)
        .def_readonly("generic_phi1_source_adjoint_allocation_count",
            &MACEKokkos<Precision>::generic_phi1_source_adjoint_allocation_count)
        .def_property_readonly("low_memory",
            &MACEKokkos<Precision>::low_memory_enabled)
        .def_property_readonly("low_memory_requested",
            &MACEKokkos<Precision>::low_memory_requested)
        .def_property_readonly("low_memory_policy",
            &MACEKokkos<Precision>::low_memory_policy_name)
        .def_property_readonly("low_memory_selection_reason",
            &MACEKokkos<Precision>::low_memory_selection_reason_name)
        .def_property_readonly("low_memory_device_free_bytes",
            &MACEKokkos<Precision>::low_memory_device_free_bytes)
        .def_property_readonly("low_memory_device_total_bytes",
            &MACEKokkos<Precision>::low_memory_device_total_bytes)
        .def_property_readonly("low_memory_reserve_bytes",
            &MACEKokkos<Precision>::low_memory_reserve_bytes)
        .def_property_readonly("low_memory_available_bytes",
            &MACEKokkos<Precision>::low_memory_available_bytes)
        .def_property_readonly("low_memory_speed_estimated_bytes",
            &MACEKokkos<Precision>::low_memory_speed_estimated_bytes)
        .def_property_readonly("low_memory_capacity_y_only_estimated_bytes",
            &MACEKokkos<Precision>::low_memory_capacity_y_only_estimated_bytes)
        .def_property_readonly("low_memory_capacity_retained_estimated_bytes",
            &MACEKokkos<Precision>::low_memory_capacity_retained_estimated_bytes)
        .def_property_readonly("low_memory_capacity_estimated_bytes",
            &MACEKokkos<Precision>::low_memory_capacity_estimated_bytes)
        .def_property_readonly("low_memory_selected_estimated_bytes",
            &MACEKokkos<Precision>::low_memory_selected_estimated_bytes)
        .def_property_readonly("execution_geometry_growth_reason",
            &MACEKokkos<Precision>::execution_geometry_growth_reason_name)
        .def_property_readonly("execution_active_receiver_count",
            &MACEKokkos<Precision>::execution_active_receiver_count)
        .def_property_readonly("execution_active_feature_node_count",
            &MACEKokkos<Precision>::execution_active_feature_node_count)
        .def_property_readonly("execution_active_edge_count",
            &MACEKokkos<Precision>::execution_active_edge_count)
        .def_property_readonly("execution_planned_receiver_capacity",
            &MACEKokkos<Precision>::execution_planned_receiver_capacity)
        .def_property_readonly("execution_planned_feature_node_capacity",
            &MACEKokkos<Precision>::execution_planned_feature_node_capacity)
        .def_property_readonly("execution_planned_edge_capacity",
            &MACEKokkos<Precision>::execution_planned_edge_capacity)
        .def_property_readonly("single_layer_workspace_active_receivers",
            &MACEKokkos<Precision>::single_layer_workspace_active_receivers)
        .def_property_readonly("single_layer_workspace_planned_receivers",
            &MACEKokkos<Precision>::single_layer_workspace_planned_receivers)
        .def_property_readonly("single_layer_workspace_active_edges",
            &MACEKokkos<Precision>::single_layer_workspace_active_edges)
        .def_property_readonly("single_layer_workspace_planned_edges",
            &MACEKokkos<Precision>::single_layer_workspace_planned_edges)
        .def_property_readonly("single_layer_workspace_bytes",
            &MACEKokkos<Precision>::single_layer_workspace_bytes)
        .def_property_readonly("single_layer_workspace_bytes_per_receiver",
            &MACEKokkos<Precision>::single_layer_workspace_bytes_per_receiver)
        .def_property_readonly("single_layer_workspace_bytes_per_edge",
            &MACEKokkos<Precision>::single_layer_workspace_bytes_per_edge)
        .def_property_readonly("single_layer_workspace_replacement_count",
            &MACEKokkos<Precision>::single_layer_workspace_replacement_count)
        .def_property_readonly("single_layer_workspace_reuse_count",
            &MACEKokkos<Precision>::single_layer_workspace_reuse_count)
        .def_property_readonly("single_layer_workspace_batch_count",
            &MACEKokkos<Precision>::single_layer_workspace_batch_count)
        .def_property_readonly("single_layer_tiled_evaluation_count",
            &MACEKokkos<Precision>::single_layer_tiled_evaluation_count)
        .def_property_readonly("dual_layer_workspace_active_receivers",
            &MACEKokkos<Precision>::dual_layer_workspace_active_receivers)
        .def_property_readonly("dual_layer_workspace_planned_receivers",
            &MACEKokkos<Precision>::dual_layer_workspace_planned_receivers)
        .def_property_readonly("dual_layer_workspace_active_edges",
            &MACEKokkos<Precision>::dual_layer_workspace_active_edges)
        .def_property_readonly("dual_layer_workspace_planned_edges",
            &MACEKokkos<Precision>::dual_layer_workspace_planned_edges)
        .def_property_readonly("dual_layer_source_segment_count",
            &MACEKokkos<Precision>::dual_layer_source_segment_count)
        .def_property_readonly("dual_layer_workspace_bytes",
            &MACEKokkos<Precision>::dual_layer_workspace_bytes)
        .def_property_readonly("dual_layer_workspace_bytes_per_receiver",
            &MACEKokkos<Precision>::dual_layer_workspace_bytes_per_receiver)
        .def_property_readonly("dual_layer_workspace_bytes_per_edge",
            &MACEKokkos<Precision>::dual_layer_workspace_bytes_per_edge)
        .def_property_readonly("dual_layer_schedule_bytes",
            &MACEKokkos<Precision>::dual_layer_schedule_bytes)
        .def_property_readonly(
            "dual_layer_schedule_preparation_explicit_scratch_bytes",
            &MACEKokkos<Precision>::
                dual_layer_schedule_preparation_explicit_scratch_bytes)
        .def_property_readonly("dual_layer_workspace_replacement_count",
            &MACEKokkos<Precision>::dual_layer_workspace_replacement_count)
        .def_property_readonly("dual_layer_workspace_reuse_count",
            &MACEKokkos<Precision>::dual_layer_workspace_reuse_count)
        .def_property_readonly("dual_layer_workspace_batch_count",
            &MACEKokkos<Precision>::dual_layer_workspace_batch_count)
        .def_property_readonly("dual_layer_tiled_evaluation_count",
            &MACEKokkos<Precision>::dual_layer_tiled_evaluation_count)
        .def_property_readonly("execution_planned_capacity_bytes",
            &MACEKokkos<Precision>::execution_planned_capacity_bytes)
        .def_property_readonly("execution_capacity_selection_reason",
            &MACEKokkos<Precision>::execution_capacity_selection_reason_name)
        .def_property_readonly("execution_result_allocation_count",
            &MACEKokkos<Precision>::execution_result_allocation_count_value)
        .def_property_readonly("edge_geometry_policy",
            &MACEKokkos<Precision>::edge_geometry_policy_name)
        .def_property_readonly("compact_edge_geometry_bytes",
            &MACEKokkos<Precision>::compact_edge_geometry_bytes)
        .def_property_readonly("harmonic_storage_policy",
            &MACEKokkos<Precision>::harmonic_storage_policy_name)
        .def_property_readonly("harmonic_storage_policy_request",
            &MACEKokkos<Precision>::harmonic_storage_policy_request_name)
        .def_property_readonly("harmonic_storage_selection_reason",
            &MACEKokkos<Precision>::harmonic_storage_selection_reason_name)
        .def_property_readonly("harmonic_storage_fallback_reason",
            &MACEKokkos<Precision>::harmonic_storage_fallback_reason_name)
        .def_property_readonly("harmonic_value_bytes",
            &MACEKokkos<Precision>::harmonic_value_bytes)
        .def_property_readonly("harmonic_gradient_bytes",
            &MACEKokkos<Precision>::harmonic_gradient_bytes)
        .def_property_readonly("shuffled_coordinate_bytes",
            &MACEKokkos<Precision>::shuffled_coordinate_bytes)
        .def_property_readonly("m1_recompute_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.m1_recompute_forward_launch_count;
            })
        .def_property_readonly("m1_recompute_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.m1_recompute_reverse_launch_count;
            })
        .def_property_readonly("macefield_response_m1_recompute_scratch_bytes",
            &MACEKokkos<Precision>::macefield_response_m1_recompute_scratch_bytes)
        .def_property_readonly(
            "macefield_response_m1_recompute_scratch_limit_bytes",
            &MACEKokkos<Precision>::
                macefield_response_m1_recompute_scratch_limit_bytes)
        .def_property_readonly(
            "macefield_response_m1_recompute_tile_channels",
            &MACEKokkos<Precision>::macefield_response_m1_recompute_tile_channels)
        .def_property_readonly(
            "macefield_response_m1_recompute_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_m1_recompute_forward_launch_count;
            })
        .def_property_readonly(
            "macefield_response_m1_recompute_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_m1_recompute_reverse_launch_count;
            })
        .def_property_readonly(
            "macefield_response_receiver_ownership",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_receiver_ownership;
            })
        .def_property_readonly(
            "macefield_response_source_ownership",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_source_ownership;
            })
        .def_property_readonly(
            "macefield_response_call_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_call_count;
            })
        .def_property_readonly(
            "macefield_response_factorized_topology_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_factorized_topology_count;
            })
        .def_property_readonly(
            "macefield_response_primal_reconstruction_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_primal_reconstruction_count;
            })
        .def_property_readonly(
            "macefield_response_phi1_fused_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_phi1_fused_launch_count;
            })
        .def_property_readonly(
            "macefield_response_phi1_generic_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_phi1_generic_launch_count;
            })
        .def_property_readonly(
            "macefield_response_a0_fused_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_a0_fused_launch_count;
            })
        .def_property_readonly(
            "macefield_response_a0_generic_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.macefield_response_a0_generic_launch_count;
            })
        .def_property_readonly(
            "streamed_edges_mode", &MACEKokkos<Precision>::streamed_edges_mode)
        .def_property_readonly(
            "supports_streamed_edges", &MACEKokkos<Precision>::supports_streamed_edges)
        .def_property_readonly(
            "supports_factorized", &MACEKokkos<Precision>::supports_factorized)
        .def_property_readonly(
            "jit_cuda_plugin_ready",
            &MACEKokkos<Precision>::jit_cuda_plugin_ready)
        .def_property_readonly(
            "jit_cuda_plugin_path",
            &MACEKokkos<Precision>::jit_cuda_plugin_path)
        .def_property_readonly(
            "jit_cuda_plugin_artifact_id",
            &MACEKokkos<Precision>::jit_cuda_plugin_artifact_id)
        .def_property_readonly(
            "jit_hip_plugin_ready",
            &MACEKokkos<Precision>::jit_hip_plugin_ready)
        .def_property_readonly(
            "jit_hip_plugin_path",
            &MACEKokkos<Precision>::jit_hip_plugin_path)
        .def_property_readonly(
            "jit_hip_plugin_artifact_id",
            &MACEKokkos<Precision>::jit_hip_plugin_artifact_id)
        .def_property_readonly(
            "jit_device_plugin_ready",
            &MACEKokkos<Precision>::jit_device_plugin_ready)
        .def_property_readonly(
            "jit_device_plugin_path",
            &MACEKokkos<Precision>::jit_device_plugin_path)
        .def_property_readonly(
            "jit_device_plugin_artifact_id",
            &MACEKokkos<Precision>::jit_device_plugin_artifact_id)
        .def_property_readonly(
            "m0_device_module_ready",
            &MACEKokkos<Precision>::m0_device_module_ready)
        .def_property_readonly(
            "m0_host_plugin_ready",
            &MACEKokkos<Precision>::m0_host_plugin_ready)
        .def_property_readonly(
            "m0_host_plugin_path",
            &MACEKokkos<Precision>::m0_host_plugin_path)
        .def_property_readonly(
            "m0_host_plugin_artifact_id",
            &MACEKokkos<Precision>::m0_host_plugin_artifact_id)
        .def_property_readonly(
            "m0_device_module_path",
            &MACEKokkos<Precision>::m0_device_module_path)
        .def_property_readonly(
            "m0_device_module_artifact_id",
            &MACEKokkos<Precision>::m0_device_module_artifact_id)
        .def_property_readonly(
            "m0_device_module_schedule",
            &MACEKokkos<Precision>::m0_device_module_schedule_name)
        .def_property_readonly(
            "r0_device_module_ready",
            &MACEKokkos<Precision>::r0_device_module_ready)
        .def_property_readonly(
            "r0_device_module_path",
            &MACEKokkos<Precision>::r0_device_module_path)
        .def_property_readonly(
            "r0_device_module_artifact_id",
            &MACEKokkos<Precision>::r0_device_module_artifact_id)
        .def_property_readonly(
            "device_cuda_environment",
            [] (const MACEKokkos<Precision>& self) {
                py::dict result;
                result["available"] = self.device_cuda_available();
                if (!self.device_cuda_available())
                    return result;
                const int compute_capability =
                    self.execution_cuda_compute_capability();
                result["backend"] = "cuda";
                result["device_name"] = self.execution_cuda_device_name();
                result["device_ordinal"] = self.execution_cuda_device_ordinal();
                result["compute_capability"] =
                    std::to_string(compute_capability/10)+"."
                    +std::to_string(compute_capability%10);
                result["compute_capability_code"] = compute_capability;
                result["multiprocessor_count"] =
                    self.execution_cuda_multiprocessor_count();
                result["warp_width"] = self.execution_cuda_warp_width();
                result["runtime_version"] = self.execution_cuda_runtime_version();
                result["driver_version"] = self.execution_cuda_driver_version();
                return result;
            })
        .def_property_readonly(
            "execution_device_execution_environment",
            [] (const MACEKokkos<Precision>&) {
                return execution_device_environment_dict();
            })
        .def_property_readonly(
            "jit_host_plugin_ready",
            &MACEKokkos<Precision>::jit_host_plugin_ready)
        .def_property_readonly(
            "jit_host_plugin_path",
            &MACEKokkos<Precision>::jit_host_plugin_path)
        .def_property_readonly(
            "jit_host_plugin_artifact_id",
            &MACEKokkos<Precision>::jit_host_plugin_artifact_id)
        .def_property_readonly("factorized_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_ready;
            })
        .def_property_readonly("factorized_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_workspace_bytes;
            })
        .def_property_readonly("factorized_workspace_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_workspace_capacity_bytes;
            })
        .def_property_readonly("factorized_schedule_build_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_schedule_build_count;
            })
        .def_property_readonly("factorized_schedule_entries",
            [] (const MACEKokkos<Precision>& self) {
                return std::max(
                    self.execution_source_edges.size(),
                    self.execution_direct_source_edges.size());
            })
        .def_property_readonly("factorized_schedule_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_schedule_bytes;
            })
        .def_property_readonly("model_load_phase_ms",
            [] (const MACEKokkos<Precision>& self) {
                auto result = py::dict();
                for (std::size_t index=0;
                     index<self.model_load_phase_names.size(); ++index)
                    result[py::str(self.model_load_phase_names[index])] =
                        self.model_load_phase_ms[index];
                return result;
            })
        .def_property_readonly("factorized_graph_generation",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_prepared_graph_generation;
            })
        .def_property_readonly("all_interactions_graph_generation",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_prepared_graph_generation;
            })
        .def_property_readonly("all_interactions_prepared_graph_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_prepared_graph_count;
            })
        .def_property_readonly("all_interactions_prepared_evaluation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_prepared_evaluation_count;
            })
        .def_property_readonly("all_interactions_geometry_update_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_geometry_update_count;
            })
        .def_property_readonly("all_interactions_schedule_build_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_schedule_build_count;
            })
        .def_property_readonly("all_interactions_active_edge_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.all_interactions_active_edge_count;
            })
        .def_property_readonly("factorized_prepared_graph_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_prepared_graph_count;
            })
        .def_property_readonly("factorized_graph_device_replacement_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_graph_device_replacement_count;
            })
        .def_property_readonly("factorized_graph_device_update_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_graph_device_update_count;
            })
        .def_property_readonly("factorized_graph_device_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_graph_device_capacity_bytes;
            })
        .def_property_readonly(
            "factorized_distributed_prefix_h1_allocation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_distributed_prefix_h1_allocation_count;
            })
        .def_property_readonly("factorized_prepared_evaluation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_prepared_evaluation_count;
            })
        .def_property_readonly("factorized_fallback_evaluation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_fallback_evaluation_count;
            })
        .def_property_readonly("factorized_topology_validation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_topology_validation_count;
            })
        .def_property_readonly("factorized_topology_validation_skip_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_topology_validation_skip_count;
            })
        .def_property_readonly("factorized_preparation_fence_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_preparation_fence_count;
            })
        .def_property_readonly("factorized_evaluation_fence_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_evaluation_fence_count;
            })
        .def_property_readonly("factorized_stage_fence_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_stage_fence_count;
            })
        .def_property_readonly("execution_geometry_capacity_edges",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_geometry_capacity_edges;
            })
        .def_property_readonly("execution_geometry_workspace_bytes",
            &MACEKokkos<Precision>::execution_geometry_workspace_bytes)
        .def_property_readonly("execution_geometry_allocation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_geometry_allocation_count;
            })
        .def_property_readonly("execution_geometry_copy_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_geometry_copy_count;
            })
        .def_property_readonly("execution_prepared_geometry_update_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_prepared_geometry_update_count;
            })
        .def_property_readonly("factorized_fractional_geometry_preparation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_fractional_geometry_preparation_count;
            })
        .def_property_readonly(
            "factorized_fractional_geometry_initialization_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_fractional_geometry_initialization_bytes;
            })
        .def_property_readonly("factorized_cell_update_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_cell_update_count;
            })
        .def_property_readonly("factorized_cell_update_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_cell_update_bytes;
            })
        .def_property_readonly("factorized_geometry_state_allocation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_geometry_state_allocation_count;
            })
        .def_property_readonly("execution_sphericart_initialization_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_sphericart_initialization_count;
            })
        .def_property_readonly("execution_sphericart_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_sphericart_launch_count;
            })
        .def_property_readonly("execution_sphericart_async_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_sphericart_async_launch_count;
            })
        .def_property_readonly("execution_direct_harmonic_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_direct_harmonic_launch_count;
            })
        .def_property_readonly("factorized_jit_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_jit_launch_count;
            })
        .def_property_readonly("factorized_jit_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_jit_forward_launch_count;
            })
        .def_property_readonly("factorized_jit_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_jit_reverse_launch_count;
            })
        .def_property_readonly("factorized_custom_blas_launch_count",
            &MACEKokkos<Precision>::factorized_custom_blas_launch_count)
        .def_property_readonly("factorized_blas_stream_bind_count",
            &MACEKokkos<Precision>::factorized_blas_stream_bind_count)
        .def_property_readonly("factorized_last_graph_prepare_ms",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_last_graph_prepare_ms;
            })
        .def_property_readonly("factorized_last_evaluation_ms",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_last_evaluation_ms;
            })
        .def_property_readonly("factorized_source_owned_reverse",
            [] (const MACEKokkos<Precision>& self) {
                return mace_uses_prepared_execution(self.streamed_edges)
                    && self.factorized_schedule_build_count > 0;
            })
        .def_property_readonly("standard_r0_module_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_module_ready;
            })
        .def_property_readonly("standard_r0_module_fallback_reason",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_module_fallback_reason;
            })
        .def_property_readonly("standard_r0_module_id",
            &MACEKokkos<Precision>::standard_r0_module_id)
        .def_property_readonly("standard_r0_module_revision",
            &MACEKokkos<Precision>::standard_r0_module_revision)
        .def_property_readonly("standard_r0_executor",
            &MACEKokkos<Precision>::standard_r0_executor_name)
        .def_property_readonly("standard_r0_selected_executor",
            &MACEKokkos<Precision>::standard_r0_selected_executor_name)
        .def_property_readonly("standard_m0_module_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_ready;
            })
        .def_property_readonly("standard_m0_has_model_contract",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_has_model_contract;
            })
        .def_property_readonly("standard_m0_model_structure_fingerprint",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_model_structure_fingerprint;
            })
        .def_property_readonly("standard_m0_module_fallback_reason",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_fallback_reason;
            })
        .def_property_readonly("standard_m0_module_id",
            &MACEKokkos<Precision>::standard_m0_module_id)
        .def_property_readonly("standard_m0_module_revision",
            &MACEKokkos<Precision>::standard_m0_module_revision)
        .def_property_readonly("standard_m0_executor",
            &MACEKokkos<Precision>::standard_m0_executor_name)
        .def_property_readonly("standard_m0_selected_executor",
            &MACEKokkos<Precision>::standard_m0_selected_executor_name)
        .def_property_readonly("r0_implementation",
            &MACEKokkos<Precision>::r0_implementation_name)
        .def_property_readonly("m0_implementation",
            &MACEKokkos<Precision>::m0_implementation_name)
        .def_property_readonly("r0_module_id",
            &MACEKokkos<Precision>::r0_selected_module_id)
        .def_property_readonly("m0_module_id",
            &MACEKokkos<Precision>::m0_selected_module_id)
        .def_property_readonly("r0_supports_low_memory",
            &MACEKokkos<Precision>::r0_supports_low_memory)
        .def_property_readonly("m0_supports_low_memory",
            &MACEKokkos<Precision>::m0_supports_low_memory)
        .def_property_readonly("standard_m0_module_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_forward_launch_count;
            })
        .def_property_readonly("m0_module_forward_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_forward_launch_count;
            })
        .def_property_readonly("standard_m0_module_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_reverse_launch_count;
            })
        .def_property_readonly("m0_module_reverse_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_module_reverse_launch_count;
            })
        .def_property_readonly("standard_m0_input_scale_adjoint_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_input_scale_adjoint_launch_count;
            })
        .def_property_readonly("m0_input_scale_adjoint_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_m0_input_scale_adjoint_launch_count;
            })
        .def_property_readonly("standard_m0_poly_values_active_bytes",
            &MACEKokkos<Precision>::standard_m0_poly_values_active_bytes)
        .def_property_readonly("standard_m0_poly_values_capacity_bytes",
            &MACEKokkos<Precision>::standard_m0_poly_values_capacity_bytes)
        .def_property_readonly("standard_m0_poly_adjoints_active_bytes",
            &MACEKokkos<Precision>::standard_m0_poly_adjoints_active_bytes)
        .def_property_readonly("standard_m0_poly_adjoints_capacity_bytes",
            &MACEKokkos<Precision>::standard_m0_poly_adjoints_capacity_bytes)
        .def_property_readonly("standard_r0_has_model_contract",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_has_model_contract;
            })
        .def_property_readonly("standard_r0_model_contract_fingerprint",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_model_contract_fingerprint;
            })
        .def_property_readonly("standard_r0_model_semantic_fingerprint",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_model_semantic_fingerprint;
            })
        .def_property_readonly("standard_r0_module_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_module_launch_count;
            })
        .def_property_readonly("r0_module_launch_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_module_launch_count;
            })
        .def_property_readonly("standard_r0_density_scale_fused",
            [] (const MACEKokkos<Precision>& self) {
                return self.standard_r0_density_scale_fused;
            })
        .def_property_readonly("standard_r0_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*(self.R0.size()+self.R0_deriv.size())
                    +sizeof(double)*(self.A0_spline_values.size()
                        +self.A0_spline_derivs.size()
                        +self.standard_r0_density_state.size());
            })
        .def_property_readonly("execution_unified_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_workspace_bytes
                    +sizeof(Precision)*(self.R0.size()+self.R0_deriv.size())
                    +sizeof(double)*(self.A0_spline_values.size()
                        +self.A0_spline_derivs.size()
                        +self.standard_r0_density_state.size());
            })
        .def_property_readonly(
            "factorized_source_strategy",
            &MACEKokkos<Precision>::factorized_source_strategy_name)
        .def_property_readonly(
            "factorized_execution_strategy",
            &MACEKokkos<Precision>::factorized_execution_strategy_name)
        .def_property_readonly(
            "factorized_execution_profile",
            &MACEKokkos<Precision>::factorized_execution_profile_name)
        .def_property_readonly(
            "factorized_derivative_signature",
            &MACEKokkos<Precision>::factorized_derivative_signature_name)
        .def_property_readonly("factorized_direct_forward_executor",
            &MACEKokkos<Precision>::factorized_direct_forward_executor_name)
        .def_property_readonly("factorized_selected_direct_forward_executor",
            &MACEKokkos<Precision>::factorized_selected_direct_forward_executor_name)
        .def_property_readonly("factorized_direct_reverse_executor",
            &MACEKokkos<Precision>::factorized_direct_reverse_executor_name)
        .def_property_readonly("factorized_selected_direct_reverse_executor",
            &MACEKokkos<Precision>::factorized_selected_direct_reverse_executor_name)
        .def_property_readonly("factorized_jit_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.jit_host_plugin_ready()
                    || self.jit_device_plugin_ready();
            })
        .def_property_readonly("factorized_jit_artifact_id",
            &MACEKokkos<Precision>::factorized_jit_artifact_id)
        .def_property_readonly("factorized_jit_contract_fingerprint",
            &MACEKokkos<Precision>::factorized_jit_contract_fingerprint)
        .def_property_readonly("execution_phi1r_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*self.Phi1r.size();
            })
        .def_property_readonly("execution_phi1r_active_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.use_factorized_direct_jit_forward()
                    ? std::size_t(0)
                    : sizeof(Precision)*self.Phi1r.size();
            })
        .def_property_readonly("execution_phi1_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*self.Phi1.size();
            })
        .def_property_readonly("execution_dphi1r_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*self.dPhi1r.size();
            })
        .def_property_readonly("execution_dphi1r_active_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.use_factorized_direct_jit_reverse()
                    ? std::size_t(0)
                    : sizeof(Precision)*self.dPhi1r.size();
            })
        .def_property_readonly("execution_dphi1_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*self.dPhi1.size();
            })
        .def_property_readonly(
            "factorized_reverse_cache_policy",
            &MACEKokkos<Precision>::factorized_reverse_cache_policy_name)
        .def_property_readonly(
            "factorized_selected_reverse_cache_policy",
            &MACEKokkos<Precision>::factorized_selected_reverse_cache_policy_name)
        .def_property_readonly(
            "factorized_retained_reverse_groups",
            &MACEKokkos<Precision>::factorized_retained_reverse_groups)
        .def_property_readonly(
            "factorized_recomputed_reverse_groups",
            &MACEKokkos<Precision>::factorized_recomputed_reverse_groups)
        .def_property_readonly("factorized_planner_budget_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_planner_budget_bytes;
            })
        .def_property_readonly("factorized_reverse_group_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_reverse_group_workspace_bytes;
            })
        .def_property_readonly("factorized_planned_coupling_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_planned_coupling_workspace_bytes;
            })
        .def_property_readonly("factorized_forward_coupling_workspace_bytes",
            &MACEKokkos<Precision>::factorized_forward_coupling_workspace_bytes)
        .def_property_readonly("factorized_reverse_coupling_workspace_bytes",
            &MACEKokkos<Precision>::factorized_reverse_coupling_workspace_bytes)
        .def_property_readonly("factorized_has_model_contract",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_has_model_contract;
            })
        .def_property_readonly("factorized_model_payload_verified",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_model_payload_verified;
            })
        .def_property_readonly("factorized_model_payload_fallback_reason",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_model_payload_fallback_reason;
            })
        .def_property_readonly("factorized_model_contract_fingerprint",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_model_contract_fingerprint;
            })
        .def_property_readonly("factorized_model_semantic_fingerprint",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_model_semantic_fingerprint;
            })
        .def_property_readonly("factorized_observer_enabled",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_observer_enabled;
            })
        .def_property_readonly("factorized_observer_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_observer_ready;
            })
        .def_property_readonly("factorized_observer_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_observer_bytes;
            })
        .def_property_readonly("execution_parameter_gradients_enabled",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_parameter_gradients_enabled;
            })
        .def_property_readonly("execution_parameter_gradients_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_parameter_gradients_ready;
            })
        .def_property_readonly("execution_parameter_gradients_result_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_parameter_gradients_result_bytes;
            })
        .def_property_readonly("execution_parameter_gradients_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.execution_parameter_gradients_workspace_bytes;
            })
        .def("set_execution_parameter_gradients",
            [] (MACEKokkos<Precision>& self,
                const bool enabled,
                const std::size_t max_bytes) {
                self.set_execution_parameter_gradients(enabled, max_bytes);
            },
            py::arg("enabled"),
            py::arg("max_bytes") = std::size_t(256)*1024*1024)
        .def("execution_parameter_gradients",
            [] (const MACEKokkos<Precision>& self) {
                const auto double_array = [] (const std::vector<double>& values) {
                    py::array_t<double> result(values.size());
                    std::copy(values.begin(), values.end(), result.mutable_data());
                    return result;
                };
                py::dict result;
                result["schema"] = "symmetrix.execution.parameter-gradients";
                result["version"] = 1;
                result["enabled"] = self.execution_parameter_gradients_enabled;
                result["ready"] = self.execution_parameter_gradients_ready;
                result["result_bytes"] =
                    self.execution_parameter_gradients_result_bytes;
                result["workspace_bytes"] =
                    self.execution_parameter_gradients_workspace_bytes;
                result["max_bytes"] = self.execution_parameter_gradients_max_bytes;
                result["owner"] = "parameter-element,receiver-order";
                result["dtype"] = "float64";
                result["parameterization"] = "extracted-compact-v2";
                result["coverage"] = "two-interaction-execution-replay";
                result["source_checkpoint_mapping"] =
                    "compact-radial-exact;H0-A0-runtime-fused";
                py::dict phase_times_ms;
                phase_times_ms["r1"] =
                    self.execution_parameter_gradients_r1_ms;
                phase_times_ms["r0"] =
                    self.execution_parameter_gradients_r0_ms;
                phase_times_ms["density"] =
                    self.execution_parameter_gradients_density_ms;
                result["phase_times_ms"] = std::move(phase_times_ms);
                py::dict worker_counts;
                worker_counts["r1"] =
                    self.execution_parameter_gradients_r1_workers;
                worker_counts["r0"] =
                    self.execution_parameter_gradients_r0_workers;
                result["worker_counts"] = std::move(worker_counts);
                py::list groups;
                for (const auto& source : self.execution_parameter_gradient_groups) {
                    py::dict group;
                    group["name"] = source.name;
                    group["layout"] = source.layout;
                    group["shape"] = source.shape;
                    group["values"] = double_array(source.values);
                    groups.append(std::move(group));
                }
                result["groups"] = std::move(groups);
                return result;
            })
        .def("_set_factorized_observer",
            [] (MACEKokkos<Precision>& self,
                const bool enabled,
                const std::size_t max_bytes) {
                self.set_factorized_observer(enabled, max_bytes);
            },
            py::arg("enabled"),
            py::arg("max_bytes") = std::size_t(256)*1024*1024)
        .def("_factorized_observer",
            [] (const MACEKokkos<Precision>& self) {
                const auto precision_array = [] (
                    const std::vector<Precision>& values) {
                    py::array_t<Precision> result(values.size());
                    std::copy(values.begin(), values.end(), result.mutable_data());
                    return result;
                };
                const auto double_array = [] (const std::vector<double>& values) {
                    py::array_t<double> result(values.size());
                    std::copy(values.begin(), values.end(), result.mutable_data());
                    return result;
                };
                py::dict result;
                result["enabled"] = self.factorized_observer_enabled;
                result["ready"] = self.factorized_observer_ready;
                result["bytes"] = self.factorized_observer_bytes;
                result["max_bytes"] = self.factorized_observer_max_bytes;
                result["num_nodes"] = self.factorized_observer_num_nodes;
                result["num_edges"] = self.factorized_observer_num_edges;
                result["embedding"] = self.factorized_embedding_width;
                result["channels"] = self.num_channels;
                result["state_layout"] =
                    "node,component,path,embedding,channel";
                py::list groups;
                for (int l=0; l<=self.l_max; ++l) {
                    const int components = 2*l+1;
                    const int paths = self.execution_group_path_offsets_host.empty()
                        ? 0 : self.execution_group_path_offsets_host[l+1]
                            -self.execution_group_path_offsets_host[l];
                    py::dict group;
                    group["l"] = l;
                    group["shape"] = py::make_tuple(
                        self.factorized_observer_num_nodes, components, paths,
                        self.factorized_embedding_width, self.num_channels);
                    if (l < static_cast<int>(self.factorized_observer_state.size())) {
                        group["state"] = precision_array(
                            self.factorized_observer_state[l]);
                        group["state_adjoint"] =
                            precision_array(
                                self.factorized_observer_state_adjoint[l]);
                    } else {
                        group["state"] = py::array_t<Precision>(0);
                        group["state_adjoint"] = py::array_t<Precision>(0);
                    }
                    groups.append(std::move(group));
                }
                result["groups"] = std::move(groups);
                result["output_layout"] = "node,harmonic,channel";
                result["output_shape"] = py::make_tuple(
                    self.factorized_observer_num_nodes,
                    self.num_lm, self.num_channels);
                result["output"] = precision_array(
                    self.factorized_observer_output);
                result["output_adjoint"] =
                    precision_array(self.factorized_observer_output_adjoint);
                result["r0_output_layout"] = "node,harmonic,channel";
                result["r0_output_shape"] = py::make_tuple(
                    self.factorized_observer_num_nodes,
                    self.num_lm, self.num_channels);
                result["r0_output"] = precision_array(
                    self.standard_r0_observer_output);
                result["r0_output_adjoint"] = precision_array(
                    self.standard_r0_observer_output_adjoint);
                result["source_h1_delta_layout"] =
                    "source,harmonic,channel";
                result["source_h1_delta_shape"] = py::make_tuple(
                    self.factorized_observer_num_nodes,
                    self.num_LM, self.num_channels);
                result["source_h1_delta"] =
                    precision_array(self.factorized_observer_source_h1_delta);
                result["directed_edge_force_delta_layout"] =
                    "directed_edge,cartesian";
                result["directed_edge_force_delta_shape"] = py::make_tuple(
                    self.factorized_observer_num_edges, 3);
                result["directed_edge_force_delta"] =
                    double_array(self.factorized_observer_directed_force_delta);
                result["r0_directed_edge_force_delta_layout"] =
                    "directed_edge,cartesian";
                result["r0_directed_edge_force_delta_shape"] = py::make_tuple(
                    self.factorized_observer_num_edges, 3);
                result["r0_directed_edge_force_delta"] =
                    double_array(self.standard_r0_observer_directed_force_delta);
                return result;
            })
        .def_property_readonly("factorized_radial_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_radial_workspace_bytes;
            })
        .def_property_readonly("factorized_arena_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return sizeof(Precision)*self.factorized_workspace_arena.size();
            })
        .def_property_readonly("factorized_arena_allocation_count",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_arena_allocation_count;
            })
        .def_property_readonly("factorized_coupling_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_coupling_workspace_bytes;
            })
        .def_property_readonly("factorized_coupling_capacity_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_coupling_capacity_bytes;
            })
        .def_property_readonly("factorized_compact_workspace_bytes",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_compact_workspace_bytes;
            })
        .def_property_readonly("factorized_tiled_ready",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_tiled_ready;
            })
        .def_property_readonly("factorized_chunk_size",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_chunk_size;
            })
        .def_property_readonly("factorized_default_chunk_size",
            [] (const MACEKokkos<Precision>&) {
                return MACEKokkos<Precision>::factorized_default_chunk_size;
            })
        .def_property_readonly("factorized_tile_selection_reason",
            [] (const MACEKokkos<Precision>& self) {
                return self.factorized_tile_selection_reason;
            })
        .def("_set_factorized_source_strategy",
            &MACEKokkos<Precision>::set_factorized_source_strategy)
        .def("_set_factorized_reverse_cache_policy",
            &MACEKokkos<Precision>::set_factorized_reverse_cache_policy)
        .def("_set_factorized_planner_budget_bytes",
            &MACEKokkos<Precision>::set_factorized_planner_budget_bytes)
        .def("_prepare_all_interactions_graph",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types) {
                prepare_active_types(self, node_types, neigh_types);
                return self.prepare_all_interactions_graph(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(
                        neigh_indices.data(), neigh_indices.size()),
                    std::span<const int>(
                        neigh_types.data(), neigh_types.size()));
            },
            py::arg("num_nodes"), py::arg("node_types"),
            py::arg("num_neigh"), py::arg("neigh_indices"),
            py::arg("neigh_types"))
        .def("_prepare_all_interactions_geometry",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t graph_generation,
                    ContiguousDoubleArray reference_positions,
                    ContiguousDoubleArray reference_xyz,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell,
                    ContiguousIntArray pbc) {
                self.prepare_all_interactions_geometry(
                    graph_generation,
                    std::span<const double>(
                        reference_positions.data(), reference_positions.size()),
                    std::span<const double>(
                        reference_xyz.data(), reference_xyz.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()),
                    std::span<const int>(pbc.data(), pbc.size()));
            },
            py::arg("graph_generation"),
            py::arg("reference_positions"), py::arg("reference_xyz"),
            py::arg("cell"), py::arg("inverse_cell"), py::arg("pbc"))
        .def("_compute_prepared_all_interactions_positions",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t graph_generation,
                    ContiguousDoubleArray positions) {
                self.compute_prepared_all_interactions_positions(
                    graph_generation,
                    std::span<const double>(positions.data(), positions.size()));
            },
            py::arg("graph_generation"), py::arg("positions"))
        .def("_prepare_factorized_graph",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types) {
                prepare_active_types(self, node_types, neigh_types);
                return self.prepare_factorized_graph(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(
                        neigh_indices.data(), neigh_indices.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()));
            },
            py::arg("num_nodes"), py::arg("node_types"),
            py::arg("num_neigh"), py::arg("neigh_indices"),
            py::arg("neigh_types"))
        .def("_prepare_periodic_factorized_graph",
            [] (MACEKokkos<Precision>& self,
                    ContiguousIntArray node_types,
                    ContiguousDoubleArray positions,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell,
                    const double neighbor_cutoff) {
                const auto [generation, num_edges] =
                    self.prepare_periodic_factorized_graph(
                        std::span<const int>(
                            node_types.data(), node_types.size()),
                        std::span<const double>(
                            positions.data(), positions.size()),
                        std::span<const double>(cell.data(), cell.size()),
                        std::span<const double>(
                            inverse_cell.data(), inverse_cell.size()),
                        neighbor_cutoff);
                py::dict result;
                result["generation"] = generation;
                result["num_edges"] = num_edges;
                return result;
            },
            py::arg("node_types"), py::arg("positions"), py::arg("cell"),
            py::arg("inverse_cell"), py::arg("neighbor_cutoff"))
        .def("_validate_graph_cardinality",
            &MACEKokkos<Precision>::validate_graph_cardinality,
            py::arg("num_receivers"), py::arg("num_feature_nodes"),
            py::arg("num_edges"))
        .def("_prepare_factorized_geometry",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray reference_positions,
                    ContiguousDoubleArray reference_xyz,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell,
                    ContiguousIntArray pbc) {
                self.prepare_factorized_geometry(
                    execution_graph_generation,
                    std::span<const double>(
                        reference_positions.data(), reference_positions.size()),
                    std::span<const double>(
                        reference_xyz.data(), reference_xyz.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()),
                    std::span<const int>(pbc.data(), pbc.size()));
            },
            py::arg("execution_graph_generation"),
            py::arg("reference_positions"), py::arg("reference_xyz"),
            py::arg("cell"), py::arg("inverse_cell"), py::arg("pbc"))
        .def("_prepare_factorized_shift_geometry",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray reference_positions,
                    ContiguousIntArray shifts,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell,
                    ContiguousIntArray pbc) {
                self.prepare_factorized_shift_geometry(
                    execution_graph_generation,
                    std::span<const double>(
                        reference_positions.data(), reference_positions.size()),
                    std::span<const int>(shifts.data(), shifts.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()),
                    std::span<const int>(pbc.data(), pbc.size()));
            },
            py::arg("execution_graph_generation"),
            py::arg("reference_positions"), py::arg("shifts"),
            py::arg("cell"), py::arg("inverse_cell"), py::arg("pbc"))
        .def("_prepare_factorized_fractional_geometry",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray fractional_positions,
                    ContiguousDoubleArray fractional_xyz,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell,
                    ContiguousIntArray pbc) {
                self.prepare_factorized_fractional_geometry(
                    execution_graph_generation,
                    std::span<const double>(
                        fractional_positions.data(), fractional_positions.size()),
                    std::span<const double>(
                        fractional_xyz.data(), fractional_xyz.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()),
                    std::span<const int>(pbc.data(), pbc.size()));
            },
            py::arg("execution_graph_generation"),
            py::arg("fractional_positions"), py::arg("fractional_xyz"),
            py::arg("cell"), py::arg("inverse_cell"), py::arg("pbc"))
        .def("_update_factorized_cell",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray cell,
                    ContiguousDoubleArray inverse_cell) {
                self.update_factorized_cell(
                    execution_graph_generation,
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const double>(
                        inverse_cell.data(), inverse_cell.size()));
            },
            py::arg("execution_graph_generation"), py::arg("cell"),
            py::arg("inverse_cell"))
        .def("_prepare_factorized_operator_benchmark",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t graph_generation,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                return self.prepare_factorized_operator_benchmark(
                    graph_generation,
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            },
            py::arg("graph_generation"), py::arg("xyz"), py::arg("r"))
        .def("_run_factorized_operator_benchmark",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t token,
                    const std::string& mode,
                    const bool synchronize) {
                self.run_factorized_operator_benchmark(token, mode);
                if (synchronize)
                    self.synchronize_factorized_operator_benchmark(token);
            },
            py::arg("token"), py::arg("mode"),
            py::arg("synchronize") = true)
        .def("_measure_factorized_operator_benchmark",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t token,
                    const std::string& mode,
                    const std::size_t warmup_iterations,
                    const double warmup_ms,
                    const std::size_t min_samples,
                    const std::size_t max_samples,
                    const double min_sample_ms) {
                const auto measurement =
                    self.measure_factorized_operator_benchmark(
                        token, mode,
                        warmup_iterations, warmup_ms,
                        min_samples, max_samples, min_sample_ms);
                auto ordered = measurement.samples_ms;
                std::sort(ordered.begin(), ordered.end());
                const auto quantile = [&] (const double q) {
                    const double index = (ordered.size()-1)*q;
                    const auto low = static_cast<std::size_t>(
                        std::floor(index));
                    const auto high = static_cast<std::size_t>(
                        std::ceil(index));
                    if (low == high)
                        return ordered[low];
                    const double weight = index-low;
                    return ordered[low]*(1.0-weight)+ordered[high]*weight;
                };
                const double total_ms = std::accumulate(
                    measurement.samples_ms.begin(),
                    measurement.samples_ms.end(), 0.0);
                const double mean_ms =
                    total_ms/measurement.samples_ms.size();
                double squared_error = 0.0;
                for (const double sample : measurement.samples_ms) {
                    const double error = sample-mean_ms;
                    squared_error += error*error;
                }
                const double std_ms = std::sqrt(
                    squared_error/measurement.samples_ms.size());
                const double median_ms = quantile(0.5);
                const int num_edges =
                    self.factorized_operator_benchmark_num_edges;

                py::dict result;
                result["mode"] = mode;
                result["cold_ms"] = measurement.cold_ms;
                result["samples_ms"] = measurement.samples_ms;
                result["median_ms"] = median_ms;
                result["mean_ms"] = mean_ms;
                result["std_ms"] = std_ms;
                result["p20_ms"] = quantile(0.2);
                result["p80_ms"] = quantile(0.8);
                result["min_ms"] = ordered.front();
                result["max_ms"] = ordered.back();
                result["sample_count"] = measurement.samples_ms.size();
                result["warmup_count"] = measurement.warmup_count;
                result["total_sample_ms"] = total_ms;
                result["ns_per_edge"] = median_ms*1.0e6
                    /std::max(1, num_edges);
                result["edges_per_second"] = median_ms > 0.0
                    ? num_edges/(median_ms/1000.0) : 0.0;
#ifdef KOKKOS_ENABLE_CUDA
                result["timing_backend"] = "cuda_events";
                result["execution_stream"] =
                    "factorized_execution_space.cuda_stream";
#else
                result["timing_backend"] = "steady_clock_with_fence";
                result["execution_stream"] =
                    "factorized_execution_space";
#endif
                py::dict config;
                config["warmup_iterations"] = warmup_iterations;
                config["warmup_ms"] = warmup_ms;
                config["min_samples"] = min_samples;
                config["max_samples"] = max_samples;
                config["min_sample_ms"] = min_sample_ms;
                config["measure_isolated_gradients"] = false;
                result["measurement_config"] = std::move(config);
#ifdef KOKKOS_ENABLE_CUDA
                const auto& device =
                    self.factorized_execution_space.cuda_device_prop();
                py::dict cuda;
                cuda["name"] = std::string(device.name);
                cuda["compute_capability"] =
                    std::to_string(device.major)+"."
                    +std::to_string(device.minor);
                cuda["multiprocessor_count"] = device.multiProcessorCount;
                int runtime_version = 0;
                int driver_version = 0;
                cudaRuntimeGetVersion(&runtime_version);
                cudaDriverGetVersion(&driver_version);
                cuda["runtime_version"] = runtime_version;
                cuda["driver_version"] = driver_version;
                result["cuda"] = std::move(cuda);
#endif
                return result;
            },
            py::arg("token"), py::arg("mode"),
            py::arg("warmup_iterations") = std::size_t(10),
            py::arg("warmup_ms") = 100.0,
            py::arg("min_samples") = std::size_t(20),
            py::arg("max_samples") = std::size_t(100),
            py::arg("min_sample_ms") = 1000.0)
        .def("_factorized_operator_benchmark_outputs",
            &factorized_operator_benchmark_outputs<Precision>,
            py::arg("token"))
        .def("_factorized_operator_export",
            &factorized_operator_export<Precision>,
            py::arg("token"))
        .def("_compute_prepared_factorized",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                self.compute_prepared_factorized(
                    execution_graph_generation,
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            },
            py::arg("execution_graph_generation"), py::arg("xyz"), py::arg("r"))
        .def("_compute_prepared_factorized_positions",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray positions) {
                self.compute_prepared_factorized_positions(
                    execution_graph_generation,
                    std::span<const double>(positions.data(), positions.size()));
            },
            py::arg("execution_graph_generation"), py::arg("positions"))
        .def("_compute_prepared_factorized_positions_field",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray positions,
                    ContiguousDoubleArray electric_field) {
                self.compute_prepared_factorized_positions_field(
                    execution_graph_generation,
                    std::span<const double>(positions.data(), positions.size()),
                    std::span<const double>(
                        electric_field.data(), electric_field.size()));
            },
            py::arg("execution_graph_generation"), py::arg("positions"),
            py::arg("electric_field"))
        .def("_compute_prepared_factorized_field",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field) {
                self.compute_prepared_factorized_field(
                    execution_graph_generation,
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()),
                    std::span<const double>(
                        electric_field.data(), electric_field.size()));
            },
            py::arg("execution_graph_generation"), py::arg("xyz"), py::arg("r"),
            py::arg("electric_field"))
        .def("_compute_prepared_factorized_field_response",
            [] (MACEKokkos<Precision>& self,
                    const std::uint64_t execution_graph_generation,
                    ContiguousDoubleArray electric_field,
                    const bool include_force_derivative) {
                self.compute_prepared_factorized_field_response(
                    execution_graph_generation,
                    create_kokkos_view("electric_field", electric_field),
                    include_force_derivative);
            },
            py::arg("execution_graph_generation"), py::arg("electric_field"),
            py::arg("include_force_derivative"))
        .def_property_readonly("scalar_size_bytes", [] (const MACEKokkos<Precision>&) {
            return sizeof(Precision);
        })
        .def_property_readonly("R0_storage_size", [] (const MACEKokkos<Precision>& self) {
            return self.R0.size()+self.R0_deriv.size();
        })
        .def_property_readonly("A0_scale_storage_size", [] (const MACEKokkos<Precision>& self) {
            return self.A0_spline_values.size()+self.A0_spline_derivs.size();
        })
        .def_property_readonly("R1_storage_size", [] (const MACEKokkos<Precision>& self) {
            return self.R1.size()+self.R1_deriv.size();
        })
        .def_property_readonly("Phi1_path_row_offsets",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.Phi1_path_row_offsets);
            })
        .def_property_readonly("atomic_numbers",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.atomic_numbers);
            })
        .def_property_readonly("atomic_energies",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.atomic_energies);
            })
        .def_readonly("selected_head", &MACEKokkos<Precision>::selected_head)
        .def_readonly("available_heads", &MACEKokkos<Precision>::available_heads)
        .def_property_readonly("active_atomic_numbers",
            [] (MACEKokkos<Precision>& self) {
                return self.active_atomic_numbers;
            })
        .def("_compact_radial_factorization_report",
            [] (MACEKokkos<Precision>& self,
                const std::string& network,
                int type_i,
                int type_j) {
                if (!self.compact_radial_model)
                    throw std::invalid_argument(
                        "Factorized radial data require a format-v2 compact model.");
                const auto report = self.compact_radial_model->factorization_report(
                    network, type_i, type_j);
                py::dict result;
                result["network"] = network;
                result["embedding_width"] = report.embedding_width;
                result["output_width"] = report.output_width;
                result["projection_layout"] = "row-major-output-by-embedding";
                result["max_value_error"] = report.max_value_error;
                result["max_derivative_error"] = report.max_derivative_error;
                return result;
            },
            py::arg("network"),
            py::arg("type_i"),
            py::arg("type_j"))
        .def("_compact_radial_backpropagation_report",
            [] (MACEKokkos<Precision>& self,
                const std::string& network,
                ContiguousIntArray pair_types_array,
                ContiguousDoubleArray coefficient_adjoints_array,
                bool factorized,
                const std::vector<double>& final_projection_adjoints) {
                if (!self.compact_radial_model)
                    throw std::invalid_argument(
                        "Compact radial backpropagation requires a format-v2 compact model.");
                if (pair_types_array.ndim() != 2
                    || pair_types_array.shape(1) != 2)
                    throw std::invalid_argument(
                        "Compact radial pair types must have shape (pair_count, 2).");

                const int pair_count = pair_types_array.shape(0);
                auto pair_types = std::vector<std::pair<int,int>>();
                pair_types.reserve(pair_count);
                for (int pair=0; pair<pair_count; ++pair)
                    pair_types.emplace_back(
                        pair_types_array.at(pair, 0),
                        pair_types_array.at(pair, 1));

                auto nodal_values = std::vector<double>();
                auto coefficient_values = std::vector<double>();
                int function_count = -1;
                const int node_count = self.compact_radial_model->num_spline_points();
                const int interval_count = node_count-1;
                const double h = self.compact_radial_model->spline_h();
                for (const auto& pair : pair_types) {
                    RadialSplineData data;
                    if (factorized) {
                        auto tables = self.compact_radial_model->materialize_factorized_pair(
                            pair.first, pair.second);
                        if (network == "R0")
                            data = std::move(tables.R0.penultimate);
                        else if (network == "R1")
                            data = std::move(tables.R1.penultimate);
                        else
                            throw std::invalid_argument(
                                "Only compact radial R0/R1 networks can be factorized.");
                    } else {
                        auto tables = self.compact_radial_model->materialize_pair(
                            pair.first, pair.second);
                        if (network == "R0")
                            data = std::move(tables.R0);
                        else if (network == "R1")
                            data = std::move(tables.R1);
                        else if (network == "A0")
                            data = std::move(tables.A0);
                        else if (network == "A1")
                            data = std::move(tables.A1);
                        else
                            throw std::invalid_argument(
                                "Compact radial network must be R0, R1, A0, or A1.");
                    }
                    if (data.values.empty())
                        throw std::invalid_argument(
                            "Requested compact radial network is absent from the model.");
                    if (function_count < 0)
                        function_count = data.values.size();
                    if (data.values.size() != static_cast<std::size_t>(function_count)
                        || data.derivatives.size()
                            != static_cast<std::size_t>(function_count))
                        throw std::runtime_error(
                            "Compact radial diagnostic has inconsistent function counts.");

                    const std::size_t nodal_offset = nodal_values.size();
                    nodal_values.resize(
                        nodal_offset+static_cast<std::size_t>(node_count)*function_count);
                    const std::size_t coefficient_offset = coefficient_values.size();
                    coefficient_values.resize(
                        coefficient_offset
                            +static_cast<std::size_t>(interval_count)*4*function_count);
                    for (int function=0; function<function_count; ++function) {
                        if (data.values[function].size()
                                != static_cast<std::size_t>(node_count)
                            || data.derivatives[function].size()
                                != static_cast<std::size_t>(node_count))
                            throw std::runtime_error(
                                "Compact radial diagnostic has inconsistent node counts.");
                        for (int node=0; node<node_count; ++node)
                            nodal_values[
                                nodal_offset+node*function_count+function] =
                                    data.values[function][node];
                        for (int interval=0; interval<interval_count; ++interval) {
                            const double value = data.values[function][interval];
                            const double next_value = data.values[function][interval+1];
                            const double derivative = data.derivatives[function][interval];
                            const double next_derivative =
                                data.derivatives[function][interval+1];
                            const std::size_t base = coefficient_offset
                                +(static_cast<std::size_t>(interval)*4)*function_count
                                +function;
                            coefficient_values[base] = value;
                            coefficient_values[base+function_count] = derivative;
                            coefficient_values[base+2*function_count] =
                                (-3.0*value-2.0*h*derivative
                                    +3.0*next_value-h*next_derivative)/(h*h);
                            coefficient_values[base+3*function_count] =
                                (2.0*value+h*derivative
                                    -2.0*next_value+h*next_derivative)/(h*h*h);
                        }
                    }
                }

                const auto coefficient_adjoints = std::vector<double>(
                    coefficient_adjoints_array.data(),
                    coefficient_adjoints_array.data()+coefficient_adjoints_array.size());
                const auto nodal_adjoints =
                    self.compact_radial_model->transpose_spline_coefficients(
                        coefficient_adjoints, pair_count, function_count);
                const auto gradients =
                    self.compact_radial_model->backpropagate_network_nodal_values(
                        network,
                        pair_types,
                        nodal_adjoints,
                        factorized,
                        final_projection_adjoints);

                const auto as_array = [] (const std::vector<double>& values) {
                    py::array_t<double> result(values.size());
                    std::copy(values.begin(), values.end(), result.mutable_data());
                    return result;
                };
                py::dict result;
                result["coefficient_layout"] =
                    "pair-interval-order-function";
                result["nodal_layout"] = "pair-node-function";
                result["function_count"] = function_count;
                result["coefficient_values"] = as_array(coefficient_values);
                result["nodal_values"] = as_array(nodal_values);
                result["nodal_adjoints"] = as_array(nodal_adjoints);
                py::list weight_gradients;
                for (const auto& layer : gradients.weights)
                    weight_gradients.append(as_array(layer));
                result["weight_gradients"] = weight_gradients;
                return result;
            },
            py::arg("network"),
            py::arg("pair_types"),
            py::arg("coefficient_adjoints"),
            py::arg("factorized") = false,
            py::arg("final_projection_adjoints") = std::vector<double>())
        .def("prepare_active_types",
            [] (MACEKokkos<Precision>& self, ContiguousIntArray node_types) {
                prepare_active_types(self, node_types);
            })
        .def_readonly("r_cut", &MACEKokkos<Precision>::r_cut)
        .def_readonly("L_max", &MACEKokkos<Precision>::L_max)
        .def_readonly(
            "single_layer_readout", &MACEKokkos<Precision>::single_layer_readout)
        // node energies
        .def_property("node_energies",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_energies);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray node_energies) {
                set_kokkos_view(self.node_energies, node_energies);
            })
        // partial forces
        .def_property("node_forces",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.node_forces);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray node_forces) {
                set_kokkos_view(self.node_forces, node_forces);
            })
        .def("_reduce_atom_forces",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray edge_receivers,
                    ContiguousIntArray edge_sources,
                    const std::uint64_t execution_graph_generation) {
                if (execution_graph_generation != 0) {
                    if (self.streamed_edges == MACEStreamedEdgesMode::generic)
                        self.reduce_prepared_all_interactions_node_forces(
                            execution_graph_generation);
                    else
                        self.reduce_prepared_node_forces(
                            execution_graph_generation);
                } else {
                    if (edge_receivers.size() != edge_sources.size())
                        throw std::invalid_argument(
                            "Atom-force reduction edge extents are inconsistent.");
                    self.reduce_node_forces(
                        num_nodes,
                        create_kokkos_view("edge_receivers", edge_receivers),
                        create_kokkos_view("edge_sources", edge_sources));
                }
                return shaped_array(
                    view_prefix_1d(
                        self.atom_forces,
                        3*static_cast<std::size_t>(num_nodes)),
                    {num_nodes, 3});
            },
            py::arg("num_nodes"), py::arg("edge_receivers"),
            py::arg("edge_sources"),
            py::arg("execution_graph_generation") = 0)
        .def("_reduce_stress",
            [] (MACEKokkos<Precision>& self,
                    const double volume,
                    ContiguousDoubleArray xyz,
                    const std::uint64_t execution_graph_generation) {
                if (execution_graph_generation != 0) {
                    if (self.streamed_edges == MACEStreamedEdgesMode::generic)
                        self.reduce_prepared_all_interactions_stress(
                            volume, execution_graph_generation);
                    else
                        self.reduce_prepared_stress(
                            volume, execution_graph_generation);
                } else {
                    self.reduce_stress(
                        volume, create_kokkos_view("stress xyz", xyz));
                }
                return shaped_array(
                    view_prefix_1d(self.stress_tensor, 9), {3, 3});
            },
            py::arg("volume"), py::arg("xyz"),
            py::arg("execution_graph_generation") = 0)
        .def_readonly("has_field_coupling", &MACEKokkos<Precision>::has_field_coupling)
        .def_property_readonly("electric_field_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_adj);
            })
        .def_property_readonly("electric_field_hessian",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_hessian);
            })
        .def_property_readonly("electric_field_force_derivative",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.electric_field_force_derivative);
            })
        // node energies and forces
        .def("compute_node_energies_forces",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    const std::uint64_t execution_graph_generation) {
                if (execution_graph_generation == 0) {
                    prepare_active_types(self, node_types, neigh_types);
                    self.compute_node_energies_forces(
                        num_nodes,
                        create_kokkos_view("node_types", node_types),
                        create_kokkos_view("num_neigh", num_neigh),
                        create_kokkos_view("neigh_indices", neigh_indices),
                        create_kokkos_view("neigh_types", neigh_types),
                        create_kokkos_view("xyz", xyz),
                        create_kokkos_view("r", r));
                    return;
                }
                self.compute_node_energies_forces(
                    num_nodes,
                    Kokkos::View<const int*>(),
                    Kokkos::View<const int*>(),
                    Kokkos::View<const int*>(),
                    Kokkos::View<const int*>(),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    execution_graph_generation);
            },
            py::arg("num_nodes"), py::arg("node_types"),
            py::arg("num_neigh"), py::arg("neigh_indices"),
            py::arg("neigh_types"), py::arg("xyz"), py::arg("r"),
            py::arg("execution_graph_generation") = 0)
        .def("compute_node_energies_forces_field",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_node_energies_forces_field(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("compute_electric_field_hessian",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_electric_field_hessian(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("compute_electric_field_force_derivative",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_electric_field_force_derivative(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("_compute_current_electric_field_hessian",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field,
                    std::uint64_t graph_generation) {
                self.compute_current_electric_field_hessian(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field),
                    graph_generation);
            })
        .def("_compute_current_electric_field_force_derivative",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    ContiguousDoubleArray electric_field,
                    std::uint64_t graph_generation) {
                self.compute_current_electric_field_force_derivative(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    create_kokkos_view("electric_field", electric_field),
                    graph_generation);
            })
        // R0
        .def_property("R0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.R0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray R0) {
                const int total_num_neigh = R0.size()/((self.l_max+1)*self.num_channels);
                set_kokkos_view(self.R0, R0, total_num_neigh, (self.l_max+1)*self.num_channels);
            })
        .def("compute_R0",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        // R1
        .def_property("R1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.R1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray R1) {
                const int num_le = self.Phi1_l.size();
                const int total_num_neigh = R1.size()/(num_le*self.num_channels);
                set_kokkos_view(self.R1, R1, total_num_neigh, num_le*self.num_channels);
            })
        .def("compute_R1",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        // Y
        .def("compute_Y",
            [] (MACEKokkos<Precision>& self,
                    ContiguousDoubleArray xyz) {
                self.compute_Y(
                    create_kokkos_view("xyz", xyz));
            })
        // A0
        .def_property("A0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A0) {
                const int num_nodes = A0.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0, A0, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A0_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A0_adj) {
                const int num_nodes = A0_adj.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A0_adj, A0_adj, num_nodes, self.num_lm, self.num_channels);
            })
        .def("compute_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_A0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_A0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                self.reverse_A0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        .def("compute_A0_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A0_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        .def("reverse_A0_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A0_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        // M0
        .def_property("M0",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M0);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M0) {
                const int num_nodes = M0.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0, M0, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("M0_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M0_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M0_adj) {
                const int num_nodes = M0_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.M0_adj, M0_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def("compute_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M0",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.reverse_M0(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H1
        .def_property("H1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray H1) {
                const int num_nodes = H1.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1, H1, num_nodes, self.num_LM, self.num_channels);
            })
        .def_property("H1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray H1_adj) {
                const int num_nodes = H1_adj.size()/(self.num_LM*self.num_channels);
                set_kokkos_view(self.H1_adj, H1_adj, num_nodes, self.num_LM, self.num_channels);
            })
        .def_readonly(
            "first_interaction_residual",
            &MACEKokkos<Precision>::first_interaction_residual)
        .def_property_readonly(
            "H1_first_residual_weights",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H1_first_residual_weights);
            })
        .def("compute_H1", &MACEKokkos<Precision>::compute_H1)
        .def("reverse_H1", &MACEKokkos<Precision>::reverse_H1)
        .def("compute_H1_product", &MACEKokkos<Precision>::compute_H1_product)
        .def("compute_H1_linear_up", &MACEKokkos<Precision>::compute_H1_linear_up)
        .def("reverse_H1_linear_up", &MACEKokkos<Precision>::reverse_H1_linear_up)
        .def("reverse_H1_product", &MACEKokkos<Precision>::reverse_H1_product)
        .def("compute_field_H1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousDoubleArray electric_field) {
                self.compute_field_H1(
                    num_nodes,
                    create_kokkos_view("electric_field", electric_field));
            })
        .def("reverse_field_H1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousDoubleArray electric_field) {
                self.reverse_field_H1(
                    num_nodes,
                    create_kokkos_view("electric_field", electric_field));
            })
        // Phi1
        .def_property("Phi1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.Phi1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray Phi1) {
                const int num_nodes = Phi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.Phi1, Phi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def_property("Phi1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.dPhi1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray dPhi1) {
                const int num_nodes = dPhi1.size()/(self.num_lme*self.num_channels);
                set_kokkos_view(self.dPhi1, dPhi1, num_nodes, self.num_lme, self.num_channels);
            })
        .def("compute_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_Phi1(
                    num_nodes,
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types));
            })
        .def("reverse_Phi1",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_indices,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r,
                    bool zero_dxyz,
                    bool zero_H1_adj) {
                self.reverse_Phi1(
                    num_nodes,
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_indices", neigh_indices),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r),
                    zero_dxyz,
                    zero_H1_adj);
            })
        // A1
        .def_property("A1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A1) {
                const int num_nodes = A1.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A1, A1, num_nodes, self.num_lm, self.num_channels);
            })
        .def_property("A1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.A1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray A1_adj) {
                const int num_nodes = A1_adj.size()/(self.num_lm*self.num_channels);
                set_kokkos_view(self.A1_adj, A1_adj, num_nodes, self.num_lm, self.num_channels);
            })
        .def("compute_A1",
            [] (MACEKokkos<Precision>& self, int num_nodes) {
                self.compute_A1(num_nodes);
            })
        .def("reverse_A1",
            [] (MACEKokkos<Precision>& self, int num_nodes) {
                self.reverse_A1(num_nodes);
            })
        .def("compute_A1_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A1_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("r", r));
            })
        .def("reverse_A1_scaled",
            [](MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A1_scaled(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    create_kokkos_view("num_neigh", num_neigh),
                    create_kokkos_view("neigh_types", neigh_types),
                    create_kokkos_view("xyz", xyz),
                    create_kokkos_view("r", r));
            })
        // M1
        .def_property("M1",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M1);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M1) {
                const int num_nodes = M1.size()/(self.num_channels);
                set_kokkos_view(self.M1, M1, num_nodes, self.num_channels);
            })
        .def_property("M1_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.M1_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousPrecisionArray M1_adj) {
                const int num_nodes = M1_adj.size()/(self.num_channels);
                set_kokkos_view(self.M1_adj, M1_adj, num_nodes, self.num_channels);
            })
        .def("compute_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_M1",
            [] (MACEKokkos<Precision>& self,
                    int num_nodes,
                    ContiguousIntArray node_types) {
                self.reverse_M1(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        // H2
        .def_property("H2",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray H2) {
                const int num_nodes = H2.size()/self.num_channels;
                set_kokkos_view(self.H2, H2, num_nodes, self.num_channels);
            })
        .def_property("H2_adj",
            [] (MACEKokkos<Precision>& self) {
                return view2vector(self.H2_adj);
            },
            [] (MACEKokkos<Precision>& self, ContiguousDoubleArray H2_adj) {
                const int num_nodes = H2_adj.size()/self.num_channels;
                set_kokkos_view(self.H2_adj, H2_adj, num_nodes, self.num_channels);
            })
        .def("compute_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types) {
                self.compute_H2(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            })
        .def("reverse_H2",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    bool zero_H1_adj) {
                self.reverse_H2(
                    num_nodes,
                    create_kokkos_view("node_types", node_types),
                    zero_H1_adj);
            })
        // readouts
        .def("compute_readouts",
            [] (MACEKokkos<Precision>& self,
                    const int num_nodes,
                    ContiguousIntArray node_types) {
                return self.compute_readouts(
                    num_nodes,
                    create_kokkos_view("node_types", node_types));
            });
}

void bind_mace_kokkos(py::module_ &m)
{
    bind_mace_kokkos<double>(m, "MACEKokkos");
    bind_mace_kokkos<float>(m, "MACEKokkosFloat");
}
