#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device_environment.hpp"

#include "mace_nonlinear_kokkos.hpp"
#include "utilities_kokkos.hpp"

namespace py=pybind11;
using IntArray=py::array_t<int,py::array::c_style|py::array::forcecast>;
using Int64Array=py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>;
using DoubleArray=py::array_t<double,py::array::c_style|py::array::forcecast>;

template<typename Evaluator>
void bind_mace_nonlinear_kokkos_evaluator(
    py::module_& module,const char* name)
{
    py::class_<Evaluator>(module,name)
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("filename"), py::arg("head") = "")
        .def_readonly("r_cut",&Evaluator::r_cut)
        .def_readonly("has_field_coupling",&Evaluator::has_field_coupling)
        .def_readonly("selected_head",&Evaluator::selected_head)
        .def_readonly("available_heads",&Evaluator::available_heads)
        .def_property_readonly("is_mh1_family",&Evaluator::is_mh1_family)
        .def_property_readonly("mh1_node_channels",&Evaluator::mh1_node_channels)
        .def_property_readonly("mh1_edge_channels",&Evaluator::mh1_edge_channels)
        .def_property_readonly("mh1_radial_size",&Evaluator::mh1_radial_size)
        .def_property_readonly("mh1_l_max",&Evaluator::mh1_l_max)
        .def_property_readonly(
            "mh1_family_rejection_reason",
            &Evaluator::mh1_family_rejection_reason)
        .def_property_readonly(
            "mh1_uses_compiled_products",
            &Evaluator::mh1_uses_compiled_products)
        .def_property_readonly(
            "mh1_uses_pair_conditioning",
            &Evaluator::mh1_uses_pair_conditioning)
        .def_property_readonly(
            "mh1_uses_external_uvu_tensors",
            &Evaluator::mh1_uses_external_uvu_tensors)
        .def_property_readonly(
            "mh1_fast_path_rejection_reason",
            &Evaluator::mh1_fast_path_rejection_reason)
        .def_property_readonly("scalar_size_bytes",[](const Evaluator&) {
            return sizeof(typename Evaluator::precision_type);
        })
        .def_property_readonly("uses_mh1_fast_path",&Evaluator::uses_mh1_fast_path)
        .def_property_readonly("supports_streamed_edges",&Evaluator::supports_streamed_edges)
        .def_property_readonly("supports_factorized",&Evaluator::supports_factorized)
        .def_property_readonly(
            "has_execution_mh1_contract",&Evaluator::has_execution_mh1_contract)
        .def_property_readonly(
            "execution_mh1_generation_fingerprint",
            &Evaluator::execution_mh1_generation_fingerprint)
        .def_property_readonly(
            "execution_mh1_semantic_fingerprint",
            &Evaluator::execution_mh1_semantic_fingerprint)
        .def_property_readonly(
            "execution_mh1_structure_fingerprint",
            &Evaluator::execution_mh1_structure_fingerprint)
        .def(
            "_set_execution_mh1_contract_identity",
            &Evaluator::set_execution_mh1_contract_identity)
        .def(
            "_set_execution_mh1_v4_contract_identity",
            &Evaluator::set_execution_mh1_v4_contract_identity)
        .def("_load_jit_mh1_host_plugin",
            &Evaluator::load_jit_mh1_host_plugin)
        .def("_load_jit_mh1_host_plugin_v4",
            &Evaluator::load_jit_mh1_host_plugin_v4)
        .def("_load_jit_mh1_host_plugin_v5",
            &Evaluator::load_jit_mh1_host_plugin_v5)
        .def_property_readonly(
            "jit_mh1_host_plugin_ready",
            &Evaluator::jit_mh1_host_plugin_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_v4_ready",
            &Evaluator::jit_mh1_host_plugin_v4_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_v5_ready",
            &Evaluator::jit_mh1_host_plugin_v5_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_path",
            &Evaluator::jit_mh1_host_plugin_path)
        .def_property_readonly(
            "jit_mh1_host_plugin_artifact_id",
            &Evaluator::jit_mh1_host_plugin_artifact_id)
        .def_property_readonly(
            "execution_mh1_retained_interaction_output_dimensions",
            &Evaluator::execution_mh1_retained_interaction_output_dimensions)
        .def("_load_jit_mh1_cuda_plugin",
            &Evaluator::load_jit_mh1_cuda_plugin)
        .def("_load_jit_mh1_cuda_plugin_v4",
            &Evaluator::load_jit_mh1_cuda_plugin_v4,
            py::arg("path"),py::arg("launch_plan_json")="")
        .def_property_readonly(
            "jit_mh1_cuda_plugin_ready",
            &Evaluator::jit_mh1_cuda_plugin_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_v4_ready",
            &Evaluator::jit_mh1_cuda_plugin_v4_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_path",
            &Evaluator::jit_mh1_cuda_plugin_path)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_artifact_id",
            &Evaluator::jit_mh1_cuda_plugin_artifact_id)
        .def_property_readonly(
            "device_cuda_environment",
            [] (const Evaluator& self) {
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
            [] (const Evaluator&) { return execution_device_environment_dict(); })
        .def_property_readonly(
            "execution_mh1_execution_backend",
            &Evaluator::execution_mh1_execution_backend)
        .def_property_readonly("streamed_edges_mode",&Evaluator::streamed_edges_mode)
        .def("set_streamed_edges",&Evaluator::set_streamed_edges)
        .def_property_readonly("mh1_edge_executor",&Evaluator::mh1_edge_executor)
        .def("set_mh1_edge_executor",&Evaluator::set_mh1_edge_executor)
        .def_property_readonly(
            "mh1_pair_spline_ready",&Evaluator::mh1_pair_spline_ready)
        .def_property_readonly(
            "mh1_pair_spline_nodes",&Evaluator::mh1_pair_spline_nodes)
        .def_property_readonly(
            "mh1_pair_spline_type_count",&Evaluator::mh1_pair_spline_type_count)
        .def(
            "_set_execution_mh1_node_arena_policy",
            &Evaluator::set_execution_mh1_node_arena_policy)
        .def_property_readonly(
            "execution_mh1_node_arena_policy",
            &Evaluator::execution_mh1_node_arena_policy)
        .def_property_readonly(
            "execution_mh1_node_arena_tile_rows",
            &Evaluator::execution_mh1_node_arena_tile_rows)
        .def(
            "_set_execution_mh1_node_arena_tile_rows_for_testing",
            &Evaluator::set_execution_mh1_node_arena_tile_rows_for_testing)
        .def_property_readonly(
            "factorized_forward_evaluation_count",
            &Evaluator::factorized_forward_evaluation_count)
        .def_property_readonly(
            "factorized_reverse_evaluation_count",
            &Evaluator::factorized_reverse_evaluation_count)
        .def_property_readonly(
            "factorized_fallback_evaluation_count",
            &Evaluator::factorized_fallback_evaluation_count)
        .def_property_readonly(
            "factorized_schedule_build_count",
            &Evaluator::factorized_schedule_build_count)
        .def_property_readonly(
            "factorized_schedule_entries",
            &Evaluator::factorized_schedule_entries)
        .def_property_readonly(
            "factorized_schedule_active_sources",
            &Evaluator::factorized_schedule_active_sources)
        .def_property_readonly(
            "factorized_schedule_bytes",
            &Evaluator::factorized_schedule_bytes)
        .def_property_readonly(
            "factorized_graph_generation",
            &Evaluator::factorized_graph_generation)
        .def_property_readonly(
            "factorized_prepared_graph_count",
            &Evaluator::factorized_prepared_graph_count)
        .def_property_readonly(
            "factorized_prepared_evaluation_count",
            &Evaluator::factorized_prepared_evaluation_count)
        .def_property_readonly(
            "factorized_topology_validation_count",
            &Evaluator::factorized_topology_validation_count)
        .def_property_readonly(
            "factorized_topology_validation_skip_count",
            &Evaluator::factorized_topology_validation_skip_count)
        .def_property_readonly(
            "factorized_preparation_fence_count",
            &Evaluator::factorized_preparation_fence_count)
        .def_property_readonly(
            "factorized_evaluation_fence_count",
            &Evaluator::factorized_evaluation_fence_count)
        .def_property_readonly(
            "factorized_stage_fence_count",
            &Evaluator::factorized_stage_fence_count)
        .def_property_readonly(
            "factorized_zbl_evaluator_stream_launch_count",
            &Evaluator::factorized_zbl_evaluator_stream_launch_count)
        .def_property_readonly(
            "execution_geometry_capacity_edges",
            &Evaluator::execution_geometry_capacity_edges)
        .def_property_readonly(
            "execution_geometry_growth_reason",
            &Evaluator::execution_geometry_growth_reason)
        .def(
            "_set_execution_geometry_device_memory_info_for_testing",
            &Evaluator::set_execution_geometry_device_memory_info_for_testing)
        .def_property_readonly(
            "execution_geometry_workspace_bytes",
            &Evaluator::execution_geometry_workspace_bytes)
        .def(
            "_set_execution_mh1_scratch_budget_bytes",
            [] (Evaluator& self,const py::ssize_t budget_bytes) {
                if(budget_bytes<0)
                    throw py::value_error(
                        "Execution MH-1 scratch budget must be non-negative.");
                self.set_execution_mh1_scratch_budget_bytes(
                    static_cast<std::size_t>(budget_bytes));
            },
            py::arg("budget_bytes"))
        .def_property_readonly(
            "execution_mh1_scratch_budget_bytes",
            &Evaluator::execution_mh1_scratch_budget_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_minimum_bytes",
            &Evaluator::execution_mh1_scratch_minimum_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_planned_bytes",
            &Evaluator::execution_mh1_scratch_planned_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_retained_bytes",
            &Evaluator::execution_mh1_scratch_retained_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_recomputed_bytes",
            &Evaluator::execution_mh1_scratch_recomputed_bytes)
        .def_property_readonly(
            "execution_mh1_scratch_recomputed_layer_count",
            &Evaluator::execution_mh1_scratch_recomputed_layer_count)
        .def_property_readonly(
            "execution_mh1_scratch_budget_satisfied",
            &Evaluator::execution_mh1_scratch_budget_satisfied)
        .def_property_readonly(
            "execution_mh1_conditioning_recomputation_count",
            &Evaluator::execution_mh1_conditioning_recomputation_count)
        .def_property_readonly(
            "execution_geometry_allocation_count",
            &Evaluator::execution_geometry_allocation_count)
        .def_property_readonly(
            "execution_geometry_copy_count",
            &Evaluator::execution_geometry_copy_count)
        .def_property_readonly(
            "execution_sphericart_initialization_count",
            &Evaluator::execution_sphericart_initialization_count)
        .def_property_readonly(
            "execution_sphericart_launch_count",
            &Evaluator::execution_sphericart_launch_count)
        .def_property_readonly(
            "execution_sphericart_async_launch_count",
            &Evaluator::execution_sphericart_async_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_forward_launch_count",
            &Evaluator::execution_mh1_generated_forward_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_source_reverse_launch_count",
            &Evaluator::execution_mh1_generated_source_reverse_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_edge_reverse_launch_count",
            &Evaluator::execution_mh1_generated_edge_reverse_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_conditioning_forward_launch_count",
            &Evaluator::execution_mh1_generated_conditioning_forward_launch_count)
        .def_property_readonly(
            "execution_mh1_generated_conditioning_reverse_launch_count",
            &Evaluator::execution_mh1_generated_conditioning_reverse_launch_count)
        .def_property_readonly(
            "factorized_source_owned_reverse",
            &Evaluator::factorized_source_owned_reverse)
        .def_property_readonly("edge_workspace_rows",&Evaluator::edge_workspace_rows)
        .def_property_readonly("edge_workspace_bytes",&Evaluator::edge_workspace_bytes)
        .def_property_readonly(
            "conditioned_mlp_workspace_bytes",
            &Evaluator::conditioned_mlp_workspace_bytes)
        .def("fence",&Evaluator::fence)
        .def("set_e3_linear_backend",&Evaluator::set_e3_linear_backend)
        .def_property_readonly("e3_linear_backend",&Evaluator::e3_linear_backend)
        .def("selected_e3_linear_backend",&Evaluator::selected_e3_linear_backend)
        .def(
            "set_execution_mh1_host_node_backend",
            &Evaluator::set_execution_mh1_host_node_backend)
        .def_property_readonly(
            "execution_mh1_host_node_backend",
            &Evaluator::execution_mh1_host_node_backend)
        .def_property_readonly(
            "selected_execution_mh1_host_node_backend",
            &Evaluator::selected_execution_mh1_host_node_backend)
        .def_property_readonly("tensor_product_backend",&Evaluator::tensor_product_backend)
        .def_property_readonly(
            "tensor_product_execution_backend",
            &Evaluator::tensor_product_execution_backend)
        .def_property_readonly(
            "tensor_product_channel_team_size",
            &Evaluator::tensor_product_channel_team_size)
        .def_property_readonly(
            "tensor_product_harmonic_team_size",
            &Evaluator::tensor_product_harmonic_team_size)
        .def(
            "set_fused_gate_normalization_reverse",
            &Evaluator::set_fused_gate_normalization_reverse)
        .def_property_readonly(
            "fused_gate_normalization_reverse_available",
            &Evaluator::fused_gate_normalization_reverse_available)
        .def_property_readonly(
            "uses_fused_gate_normalization_reverse",
            &Evaluator::uses_fused_gate_normalization_reverse)
        .def(
            "set_direct_node_tensor_reverse",
            &Evaluator::set_direct_node_tensor_reverse)
        .def_property_readonly(
            "direct_node_tensor_reverse_available",
            &Evaluator::direct_node_tensor_reverse_available)
        .def_property_readonly(
            "uses_direct_node_tensor_reverse",
            &Evaluator::uses_direct_node_tensor_reverse)
        .def_property_readonly("linear_workspace_bytes",&Evaluator::linear_workspace_bytes)
        .def_property_readonly("tensor_workspace_bytes",&Evaluator::tensor_workspace_bytes)
        .def_property_readonly("product_workspace_bytes",&Evaluator::product_workspace_bytes)
        .def_property_readonly("node_workspace_bytes",&Evaluator::node_workspace_bytes)
        .def_property_readonly("precision_workspace_bytes",&Evaluator::precision_workspace_bytes)
        .def_property_readonly("atomic_numbers",[](Evaluator& self){return self.atomic_numbers_host;})
        .def_property_readonly("atomic_energies",[](Evaluator& self){return view2vector(self.atomic_energies);})
        .def_property_readonly("node_energies",[](Evaluator& self){return view2vector(self.node_energies);})
        .def_property_readonly("node_forces",[](Evaluator& self){return view2vector(self.node_forces);})
        .def("_prepare_factorized_graph",
            [] (Evaluator& self,const int num_nodes,IntArray node_types,
                    IntArray num_neigh,IntArray neigh_indices,
                    IntArray neigh_types) {
                return self.prepare_factorized_graph(
                    num_nodes,
                    std::span<const int>(node_types.data(),node_types.size()),
                    std::span<const int>(num_neigh.data(),num_neigh.size()),
                    std::span<const int>(
                        neigh_indices.data(),neigh_indices.size()),
                    std::span<const int>(neigh_types.data(),neigh_types.size()));
            },
            py::arg("num_nodes"),py::arg("node_types"),
            py::arg("num_neigh"),py::arg("neigh_indices"),
            py::arg("neigh_types"))
        .def("_prepare_factorized_geometry",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray reference_positions,DoubleArray reference_xyz,
                    DoubleArray cell,DoubleArray inverse_cell,IntArray pbc) {
                self.prepare_factorized_geometry(
                    graph_generation,
                    std::span<const double>(
                        reference_positions.data(),reference_positions.size()),
                    std::span<const double>(
                        reference_xyz.data(),reference_xyz.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()),
                    std::span<const int>(pbc.data(),pbc.size()));
            },
            py::arg("graph_generation"),py::arg("reference_positions"),
            py::arg("reference_xyz"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("pbc"))
        .def("_compute_prepared_factorized",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray xyz,DoubleArray distances) {
                self.compute_prepared_factorized(
                    graph_generation,
                    std::span<const double>(xyz.data(),xyz.size()),
                    std::span<const double>(
                        distances.data(),distances.size()));
            },
            py::arg("graph_generation"),py::arg("xyz"),
            py::arg("distances"))
        .def("_compute_prepared_factorized_positions",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    DoubleArray positions) {
                self.compute_prepared_factorized_positions(
                    graph_generation,
                    std::span<const double>(positions.data(),positions.size()));
            },
            py::arg("graph_generation"),py::arg("positions"))
        .def("_reduce_atom_forces",
            [] (Evaluator& self,const int num_nodes,IntArray edge_receivers,
                    IntArray edge_sources,const std::uint64_t graph_generation) {
                if(num_nodes<0
                    ||edge_receivers.size()!=edge_sources.size())
                    throw py::value_error(
                        "MH-1 atom-force reduction extents are inconsistent.");
                if(graph_generation!=0)
                    self.reduce_prepared_node_forces(graph_generation);
                else
                    self.reduce_node_forces(
                        num_nodes,
                        create_kokkos_view(
                            "MH-1 force reduction receivers",edge_receivers),
                        create_kokkos_view(
                            "MH-1 force reduction sources",edge_sources));
                if(self.atom_forces.size()
                    !=3*static_cast<std::size_t>(num_nodes))
                    throw py::value_error(
                        "MH-1 atom-force reduction node extent is inconsistent.");
                return view2vector(self.atom_forces);
            },
            py::arg("num_nodes"),py::arg("edge_receivers"),
            py::arg("edge_sources"),py::arg("graph_generation")=0)
        .def("_reduce_stress",
            [] (Evaluator& self,const double volume,DoubleArray xyz,
                    const std::uint64_t graph_generation) {
                if(graph_generation!=0)
                    self.reduce_prepared_stress(volume,graph_generation);
                else
                    self.reduce_stress(
                        volume,create_kokkos_view("MH-1 stress xyz",xyz));
                if(self.stress_tensor.size()!=9)
                    throw py::value_error(
                        "MH-1 stress reduction tensor extent is inconsistent.");
                return view2vector(self.stress_tensor);
            },
            py::arg("volume"),py::arg("xyz"),
            py::arg("graph_generation")=0)
        .def("_prepare_factorized_batch",
            [] (Evaluator& self,const std::uint64_t graph_generation,
                    Int64Array edge_offsets) {
                self.prepare_factorized_batch(
                    graph_generation,
                    std::span<const std::int64_t>(
                        edge_offsets.data(),edge_offsets.size()));
            },
            py::arg("graph_generation"),py::arg("edge_offsets"))
        .def("_reduce_batched_stress",
            [] (Evaluator& self,DoubleArray volumes,
                    const std::uint64_t graph_generation) {
                self.reduce_prepared_batched_stress(
                    std::span<const double>(volumes.data(),volumes.size()),
                    graph_generation);
                if(self.stress_tensor.size()!=9*volumes.size())
                    throw py::value_error(
                        "MH-1 batched stress tensor extent is inconsistent.");
                return view2vector(self.stress_tensor);
            },
            py::arg("volumes"),py::arg("graph_generation"))
        .def("compute_node_energies_forces",
            [](Evaluator& self,int num_nodes,IntArray node_types,
                    IntArray num_neigh,IntArray neigh_indices,
                    IntArray neigh_types,DoubleArray xyz,
                    DoubleArray distances) {
                try {
                    self.compute_node_energies_forces(
                        num_nodes,
                        create_kokkos_view("nonlinear node types",node_types),
                        create_kokkos_view("nonlinear num neigh",num_neigh),
                        create_kokkos_view("nonlinear neigh indices",neigh_indices),
                        create_kokkos_view("nonlinear neigh types",neigh_types),
                        create_kokkos_view("nonlinear xyz",xyz),
                        create_kokkos_view("nonlinear distances",distances));
                } catch (...) {
                    self.fence();
                    throw;
                }
            });
}

void bind_mace_nonlinear_kokkos(py::module_& module)
{
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearKokkos>(
        module,"MACENonlinearKokkos");
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearFloatKokkos>(
        module,"MACENonlinearKokkosFloat");
}
