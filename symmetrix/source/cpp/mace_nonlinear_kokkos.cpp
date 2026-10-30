#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "device_environment.hpp"

#include "family_interface.hpp"
#include "implementation_registry.hpp"
#include "mace_nonlinear_kokkos.hpp"
#include "mace_polar_binding.hpp"
#include "mace_polar_delta_binding.hpp"
#include "mace_polar_deltas.hpp"
#include "mace_polar_kokkos.hpp"
#include "mace_polar_schema.hpp"
#include "utilities_kokkos.hpp"

namespace py=pybind11;
using IntArray=py::array_t<int,py::array::c_style|py::array::forcecast>;
using Int64Array=py::array_t<std::int64_t,py::array::c_style|py::array::forcecast>;
using DoubleArray=py::array_t<double,py::array::c_style|py::array::forcecast>;

namespace {

// The neutral artifact envelope as the dict every readiness binding reports, so
// the readiness property and the artifact loader cannot drift apart.
py::dict family_artifact_readiness_dict(
    const symmetrix::execution::ArtifactReadiness& readiness)
{
    py::dict result;
    result["ready"] = readiness.ready;
    result["target"] = readiness.target;
    result["artifact_id"] = readiness.artifact_id;
    result["generation_fingerprint"] = readiness.generation_fingerprint;
    result["semantic_fingerprint"] = readiness.semantic_fingerprint;
    result["structure_fingerprint"] = readiness.structure_fingerprint;
    result["reason"] = readiness.reason;
    return result;
}

py::list mh1_irrep_block_pattern_list(
    const std::vector<MaceMH1IrrepBlockPattern>& blocks)
{
    py::list result;
    for (const auto& block : blocks) {
        py::dict entry;
        entry["multiplicity"] = block.multiplicity;
        entry["l"] = block.l;
        entry["parity"] = block.parity;
        result.append(std::move(entry));
    }
    return result;
}

py::dict mh1_layer_descriptor_dict(const MaceMH1LayerDescriptor& layer)
{
    py::dict result;
    result["interaction_index"] = layer.interaction_index;
    result["uvu_channels"] = layer.uvu_channels;
    result["edge_blocks"] = mh1_irrep_block_pattern_list(layer.edge_blocks);
    result["product_output_blocks"] =
        mh1_irrep_block_pattern_list(layer.product_output_blocks);
    result["final_output_policy"] =
        mace_mh1_final_output_policy_name(layer.final_output_policy);
    return result;
}

py::dict mh1_family_descriptor_dict(
    const MaceMH1FamilyDescriptor& descriptor)
{
    py::dict result;
    result["compatible"] = descriptor.compatible;
    result["l_max"] = descriptor.l_max;
    result["radial_size"] = descriptor.radial_size;
    result["node_channels"] = descriptor.node_channels;
    result["first_layer_edge_channels"] =
        descriptor.first_layer_edge_channels();
    result["rejection_reason"] = descriptor.rejection_reason;
    py::list layers;
    for (const auto& layer : descriptor.layers)
        layers.append(mh1_layer_descriptor_dict(layer));
    result["layers"] = std::move(layers);
    return result;
}

// The MH-1 stage description rendered in the contract's own schedule
// vocabulary, so a test can compare the native description with the schedule
// the contract generates for the same layer count instead of trusting that the
// two vocabularies still agree.
py::dict stage_description_dict(
    const symmetrix::mace_nonlinear::StageDescription& description)
{
    auto stage_entry = [](const symmetrix::mace_nonlinear::Stage& stage) {
        py::dict entry;
        switch (stage.kind) {
        case symmetrix::mace_nonlinear::StageKind::node_program:
            entry["kind"] = "node_program";
            entry["phase"] = stage.phase;
            break;
        case symmetrix::mace_nonlinear::StageKind::edge_tensor_product:
            entry["kind"] = "edge_tensor_product";
            entry["layer"] = stage.layer;
            break;
        case symmetrix::mace_nonlinear::StageKind::edge_tensor_product_reverse:
            entry["kind"] = "edge_tensor_product_reverse";
            entry["layer"] = stage.layer;
            break;
        }
        return entry;
    };
    py::dict result;
    result["layer_count"] = description.layer_count();
    auto forward = py::list();
    for (const auto& stage : description.forward_schedule())
        forward.append(stage_entry(stage));
    auto reverse = py::list();
    for (const auto& stage : description.reverse_schedule())
        reverse.append(stage_entry(stage));
    result["forward_schedule"] = forward;
    result["reverse_schedule"] = reverse;
    return result;
}

} // namespace


template<typename Evaluator>
py::class_<Evaluator> bind_mace_nonlinear_kokkos_evaluator(
    py::module_& module,const char* name)
{
    py::class_<Evaluator> evaluator(module,name);
    evaluator
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("filename"), py::arg("head") = "")
        .def_readonly("r_cut",&Evaluator::r_cut)
        .def_readonly("has_field_coupling",&Evaluator::has_field_coupling)
        .def_readonly("selected_head",&Evaluator::selected_head)
        .def_readonly("available_heads",&Evaluator::available_heads)
        .def_property_readonly("is_mh1_family",&Evaluator::is_mh1_family)
        .def_property_readonly("mh1_node_channels",&Evaluator::mh1_node_channels)
        .def_property_readonly("mh1_layer_count",&Evaluator::mh1_layer_count)
        .def(
            "mh1_layer_descriptor",
            [](const Evaluator& self, const std::size_t index) {
                return mh1_layer_descriptor_dict(
                    self.mh1_layer_descriptor(index));
            },
            py::arg("index"))
        .def(
            "mh1_layer_retained_feature_layout",
            [](const Evaluator& self, const std::size_t index) {
                const auto& layout=
                    self.mh1_layer_retained_feature_layout(index);
                py::dict result;
                result["width"] = layout.width;
                result["channels"] = layout.channels;
                result["scalar_offset"] = layout.scalar_offset;
                result["scalar_multiplicity"] = layout.scalar_multiplicity;
                result["vector_offset"] = layout.vector_offset;
                result["vector_multiplicity"] = layout.vector_multiplicity;
                result["vector_components"] = layout.vector_components;
                result["retained_from_native_components"] =
                    layout.retained_from_native_components;
                result["retained_from_native_signs"] =
                    layout.retained_from_native_signs;
                result["portable_storage"] = "mul_ir";
                result["generated_storage"] = "ir_mul";
                return result;
            },
            py::arg("index"))
        .def_property_readonly(
            "mh1_first_layer_edge_channels",
            &Evaluator::mh1_first_layer_edge_channels)
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
        .def_property_readonly(
            "mh1_portable_diagnostics",
            [] (const Evaluator& self) {
                py::dict result;
                result["portable_executor"] = self.mh1_edge_executor();
                py::list layers;
                for(std::size_t index=0;index<self.mh1_layer_count();++index) {
                    const auto& descriptor=self.mh1_layer_descriptor(index);
                    py::dict layer;
                    layer["interaction_index"] = descriptor.interaction_index;
                    layer["uvu_channels"] = descriptor.uvu_channels;
                    layer["architecture_admitted"] = self.is_mh1_family();
                    layer["tensor_product_ready"] =
                        self.mh1_layer_tensor_product_ready(index);
                    layer["compiled_product_ready"] =
                        self.mh1_layer_compiled_product_ready(index);
                    layer["pair_conditioning_ready"] =
                        self.mh1_layer_pair_conditioning_ready(index);
                    layer["portable_ready"] =
                        self.mh1_layer_portable_ready(index);
                    layer["reason"] = self.mh1_layer_capability_reason(index);
                    layers.append(std::move(layer));
                }
                result["layers"] = std::move(layers);
                return result;
            })
        .def_property_readonly("scalar_size_bytes",[](const Evaluator&) {
            return sizeof(typename Evaluator::precision_type);
        })
        .def_property_readonly("uses_mh1_fast_path",&Evaluator::uses_mh1_fast_path)
        .def_property_readonly(
            "mh1_portable_final_layer_ready",
            &Evaluator::mh1_portable_final_layer_ready)
        .def_property_readonly(
            "mh1_portable_final_layer_enabled",
            &Evaluator::mh1_portable_final_layer_enabled)
        .def_property_readonly(
            "mh1_portable_final_layer_kind",
            &Evaluator::mh1_portable_final_layer_kind)
        .def_property_readonly(
            "mh1_portable_final_layer_evaluation_count",
            &Evaluator::mh1_portable_final_layer_evaluation_count)
        .def("_set_mh1_portable_final_layer",
            &Evaluator::set_mh1_portable_final_layer,
            py::arg("enabled"))
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
        .def("_load_jit_mh1_host_plugin_v6",
            &Evaluator::load_jit_mh1_host_plugin_v6)
        .def("_load_jit_mh1_host_plugin_v7",
            &Evaluator::load_jit_mh1_host_plugin_v7)
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
            "jit_mh1_host_plugin_v6_ready",
            &Evaluator::jit_mh1_host_plugin_v6_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_v7_ready",
            &Evaluator::jit_mh1_host_plugin_v7_ready)
        .def_property_readonly(
            "jit_mh1_host_plugin_path",
            &Evaluator::jit_mh1_host_plugin_path)
        .def_property_readonly(
            "jit_mh1_host_plugin_artifact_id",
            &Evaluator::jit_mh1_host_plugin_artifact_id)
        .def_property_readonly(
            "execution_mh1_retained_interaction_output_dimensions",
            &Evaluator::execution_mh1_retained_interaction_output_dimensions)
        .def_property_readonly(
            "execution_mh1_retained_interaction_output_bytes",
            &Evaluator::execution_mh1_retained_interaction_output_bytes)
        .def_property_readonly(
            "execution_mh1_retained_pre_gate_bytes",
            &Evaluator::execution_mh1_retained_pre_gate_bytes)
        .def_property_readonly(
            "execution_mh1_product_tile_scratch_bytes",
            &Evaluator::execution_mh1_product_tile_scratch_bytes)
        .def_property_readonly(
            "execution_mh1_layer_up_adjoint_storage_bytes",
            &Evaluator::execution_mh1_layer_up_adjoint_storage_bytes)
        .def_property_readonly(
            "execution_mh1_up_adjoint_scratch_bytes",
            &Evaluator::execution_mh1_up_adjoint_scratch_bytes)
        .def_property_readonly(
            "execution_mh1_up_adjoint_scratch_reuse_count",
            &Evaluator::execution_mh1_up_adjoint_scratch_reuse_count)
        .def("_load_jit_mh1_cuda_plugin",
            &Evaluator::load_jit_mh1_cuda_plugin)
        .def("_load_jit_mh1_cuda_plugin_v4",
            &Evaluator::load_jit_mh1_cuda_plugin_v4,
            py::arg("path"),py::arg("launch_plan_json")="")
        .def("_load_jit_mh1_cuda_plugin_v5",
            &Evaluator::load_jit_mh1_cuda_plugin_v5,
            py::arg("path"),py::arg("launch_plan_json"))
        .def_property_readonly(
            "jit_mh1_cuda_plugin_ready",
            &Evaluator::jit_mh1_cuda_plugin_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_v4_ready",
            &Evaluator::jit_mh1_cuda_plugin_v4_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_v5_ready",
            &Evaluator::jit_mh1_cuda_plugin_v5_ready)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_path",
            &Evaluator::jit_mh1_cuda_plugin_path)
        .def_property_readonly(
            "jit_mh1_cuda_plugin_artifact_id",
            &Evaluator::jit_mh1_cuda_plugin_artifact_id)
        .def(
            "_load_execution_mh1_product_module",
            &Evaluator::load_execution_mh1_product_module,
            py::arg("layer"),py::arg("kind"),py::arg("path"),
            py::arg("metadata_json"),py::arg("schedule"),
            py::arg("persistent_blocks"),py::arg("backend"))
        .def(
            "_execution_mh1_product_module_ready",
            &Evaluator::execution_mh1_product_module_ready)
        .def_property_readonly(
            "execution_mh1_product_modules_ready",
            &Evaluator::execution_mh1_product_modules_ready)
        .def(
            "_clear_execution_mh1_product_modules",
            &Evaluator::clear_execution_mh1_product_modules)
        .def(
            "_execution_mh1_product_module_artifact_id",
            &Evaluator::execution_mh1_product_module_artifact_id)
        .def(
            "_execution_mh1_product_module_forward_launch_count",
            &Evaluator::execution_mh1_product_module_forward_launch_count)
        .def(
            "_execution_mh1_product_module_reverse_launch_count",
            &Evaluator::execution_mh1_product_module_reverse_launch_count)
        .def(
            "_test_execution_mh1_product_module_launch_validation",
            &Evaluator::test_execution_mh1_product_module_launch_validation,
            py::arg("layer"),py::arg("reverse"),py::arg("num_nodes"),
            py::arg("alias_input")=true)
        .def_property_readonly(
            "execution_mh1_product_module_diagnostics",
            &Evaluator::execution_mh1_product_module_diagnostics)
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
        .def("set_retain_layer_features",&Evaluator::set_retain_layer_features,
            py::arg("retain"),
            "Retain the per-layer node features of the next forward passes: the "
            "product outputs the model's own per-layer readouts consume.  The "
            "feature mixer and the spin-source heads of the polar family read "
            "exactly these, so the retention is what gives the chain's input a "
            "producer on the evaluator's surface.  Retaining copies each "
            "layer's output, so it is opt-in.")
        .def_property_readonly("retains_layer_features",
            &Evaluator::retains_layer_features)
        .def("layer_features",
            [] (const Evaluator& self) {
                py::list result;
                for (const auto& layer : self.layer_features()) {
                    const auto mirror=Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(),layer);
                    py::array_t<double> array(std::vector<py::ssize_t>{
                        static_cast<py::ssize_t>(mirror.extent(0)),
                        static_cast<py::ssize_t>(mirror.extent(1))});
                    double* data=array.mutable_data();
                    for(std::size_t row=0;row<mirror.extent(0);++row)
                        for(std::size_t column=0;column<mirror.extent(1);++column)
                            data[row*mirror.extent(1)+column]=
                                static_cast<double>(mirror(row,column));
                    result.append(std::move(array));
                }
                return result;
            },
            "The retained per-layer node features, in layer order ([nodes, "
            "hidden width] each); empty when the retention is off or no "
            "forward pass has run since it was enabled.")
        .def("set_layer_product_seed",
            [] (Evaluator& self,py::list seeds) {
                using Precision=typename Evaluator::precision_type;
                using SeedView=Kokkos::View<
                    const Precision**,Kokkos::LayoutRight>;
                std::vector<SeedView> views;
                views.reserve(seeds.size());
                // The arrays are copied into the evaluator's own storage, so
                // the caller's buffers do not have to outlive the call.
                for (const auto& item : seeds) {
                    const auto array=py::array_t<double,py::array::c_style
                        |py::array::forcecast>::ensure(item);
                    if (!array || array.ndim() != 2)
                        throw py::value_error(
                            "MH-1 layer-product seeds must be [nodes, width] "
                            "float64 arrays.");
                    const py::ssize_t rows=array.shape(0);
                    const py::ssize_t columns=array.shape(1);
                    Kokkos::View<Precision**,Kokkos::LayoutRight> storage(
                        "MH-1 layer product seed",rows,columns);
                    auto mirror=Kokkos::create_mirror_view(storage);
                    for(py::ssize_t row=0;row<rows;++row)
                        for(py::ssize_t column=0;column<columns;++column)
                            mirror(row,column)=static_cast<Precision>(
                                array.at(row,column));
                    Kokkos::deep_copy(storage,mirror);
                    views.emplace_back(storage);
                }
                self.set_layer_product_seed(views);
            },
            py::arg("seeds"),
            "Adopt one product seed per interaction layer, in layer order: the "
            "consumer's derivative of that layer's *retained* node features "
            "(the reverse tap of ``set_retain_layer_features``).  The reverse "
            "adds each seed, converted back to the product basis, into the "
            "layer's product adjoint where the model's own per-layer readouts "
            "enter, so a consumer that reads the retained features takes its "
            "share of the backbone's own gradient and of every cross term "
            "through them.  An empty list clears.")
        .def("clear_layer_product_seed",&Evaluator::clear_layer_product_seed,
            "Drop the adopted per-layer product seeds (the next reverse is "
            "unseeded).")
        .def_property_readonly("has_layer_product_seed",
            &Evaluator::has_layer_product_seed)
        .def("layer_feature_components",&Evaluator::layer_feature_components,
            py::arg("width"),
            "The number of spherical-harmonic components per node channel of a "
            "layer whose node-feature width is ``width``, or an error naming the "
            "basis this build cannot convert (only l<=1 node features have a "
            "model-convention conversion).  The retention and the reverse tap "
            "share the rule, so a family that has to refuse such a basis before "
            "its first forward asks here.")
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
            "factorized_device_schedule_build_count",
            &Evaluator::factorized_device_schedule_build_count)
        .def_property_readonly(
            "factorized_schedule_topology_d2h_elements",
            &Evaluator::factorized_schedule_topology_d2h_elements)
        .def_property_readonly(
            "factorized_prepared_host_topology_elements",
            &Evaluator::factorized_prepared_host_topology_elements)
        .def_property_readonly(
            "factorized_schedule_host_topology_elements",
            &Evaluator::factorized_schedule_host_topology_elements)
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
        .def("_prepare_periodic_factorized_graph",
            [] (Evaluator& self,IntArray node_types,DoubleArray positions,
                    DoubleArray cell,DoubleArray inverse_cell,
                    const double neighbor_cutoff) {
                const auto [generation,num_edges]=
                    self.prepare_periodic_factorized_graph(
                        std::span<const int>(
                            node_types.data(),node_types.size()),
                        std::span<const double>(
                            positions.data(),positions.size()),
                        std::span<const double>(cell.data(),cell.size()),
                        std::span<const double>(
                            inverse_cell.data(),inverse_cell.size()),
                        neighbor_cutoff);
                py::dict result;
                result["generation"]=generation;
                result["num_edges"]=num_edges;
                return result;
            },
            py::arg("node_types"),py::arg("positions"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("neighbor_cutoff"))
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
        .def("_arm_factorized_geometry_prepare_failure",
            &Evaluator::arm_factorized_geometry_prepare_failure_for_testing,
            "Inject one failure after staged geometry initialization but "
            "before prepared-state publication.")
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
            })
        // --- Family adapter interface ---------------------------------------
        .def_property_readonly(
            "family_capabilities",
            [] (const Evaluator& self) {
                const auto capabilities = self.family_capabilities();
                py::dict result;
                result["target"] = capabilities.target;
                result["precision"] = capabilities.precision;
                result["reference"] = capabilities.reference_ready;
                result["portable"] = capabilities.portable_ready;
                result["generated"] = capabilities.generated_ready;
                result["stress"] = capabilities.stress;
                result["field"] = capabilities.field;
                result["parameter_gradients"] =
                    capabilities.parameter_gradients;
                result["host_artifact"] = capabilities.loads_host_artifact;
                result["device_artifact"] = capabilities.loads_device_artifact;
                return result;
            })
        .def_property_readonly("family_program",
            [] (const Evaluator& self) {
                const auto program = self.compile_program();
                const auto buffer_dict = [] (
                    const symmetrix::execution::BufferContract& buffer) {
                    py::dict entry;
                    entry["name"] = buffer.name;
                    entry["dtype"] = buffer.dtype;
                    py::list shape;
                    for (const auto size : buffer.shape)
                        shape.append(size);
                    entry["shape"] = shape;
                    entry["layout"] = buffer.layout.order;
                    entry["irreps"] = buffer.layout.irreps;
                    return entry;
                };
                py::dict result;
                result["family"] = program.family;
                result["precision"] = program.precision;
                result["version"] = program.version;
                result["fingerprint"] = program.fingerprint;
                py::list operators;
                for (const auto& node : program.operators) {
                    py::dict entry;
                    // The neutral node-program enum is an internal identity
                    // split.  Keep the established Python contract spelling,
                    // which also matches its legacy fingerprint token.
                    entry["kind"] = node.kind
                            == symmetrix::execution::OperatorKind::node_program
                        ? "custom"
                        : symmetrix::execution::operator_kind_name(node);
                    entry["id"] = node.id;
                    entry["capability_key"] = node.capability_key;
                    py::dict parameters;
                    for (const auto& [name, value] : node.parameters)
                        parameters[py::cast(name)] = value;
                    entry["parameters"] = parameters;
                    py::list inputs;
                    for (const auto& input : node.inputs)
                        inputs.append(buffer_dict(input));
                    entry["inputs"] = inputs;
                    py::list outputs;
                    for (const auto& output : node.outputs)
                        outputs.append(buffer_dict(output));
                    entry["outputs"] = outputs;
                    operators.append(entry);
                }
                result["operators"] = operators;
                return result;
            })
        .def_property_readonly(
            "family_binding_report",
            [] (const Evaluator& self) {
                const auto report = self.family_binding_report();
                py::dict result;
                result["version"] = report.version;
                result["target"] = report.target;
                result["precision"] = report.precision;
                result["artifact_ready"] = report.artifact_ready;
                result["admitted"] = report.admitted;
                result["node_count"] = report.node_count;
                result["portable_node_count"] = report.portable_node_count;
                py::list bindings;
                for (const auto& binding : report.bindings) {
                    py::dict entry;
                    entry["operator_id"] = binding.operator_id;
                    entry["kind"] = binding.kind
                            == symmetrix::execution::OperatorKind::node_program
                        ? "custom"
                        : !binding.custom_kind.empty()
                        ? binding.custom_kind
                        : symmetrix::execution::operator_kind_name(binding.kind);
                    entry["capability_key"] = binding.capability_key;
                    entry["implementation_id"] = binding.implementation_id;
                    entry["selected_implementation_id"] =
                        binding.selected_implementation_id;
                    entry["tier"] = symmetrix::execution::implementation_tier_name(
                        binding.tier);
                    entry["target"] = binding.target;
                    py::dict precision;
                    precision["compute"] = binding.precision.compute;
                    precision["accumulator"] = binding.precision.accumulator;
                    entry["precision"] = precision;
                    entry["identity_fingerprint"] = binding.identity_fingerprint;
                    entry["qualified"] = binding.qualified;
                    entry["portable_covered"] = binding.portable_covered;
                    entry["artifact_required"] = binding.artifact_required;
                    entry["reason"] = binding.reason;
                    bindings.append(entry);
                }
                result["bindings"] = bindings;
                return result;
            })
        .def_property_readonly(
            "family_artifact_ready",
            [] (const Evaluator& self) {
                return self.family_artifact_ready();
            })
        .def_property_readonly(
            "family_artifact_readiness",
            [] (const Evaluator& self) {
                return family_artifact_readiness_dict(
                    self.family_artifact_readiness());
            })
        .def_property_readonly(
            "family_diagnostics",
            [] (const Evaluator& self) {
                const auto diagnostics = self.family_diagnostics();
                py::dict result;
                result["fallback_count"] = diagnostics.fallback_count;
                result["diagnostic"] = diagnostics.diagnostic;
                return result;
            })
        .def("load_family_artifact",
            [] (Evaluator& self,const std::string& path,
                    const std::string& target) {
                symmetrix::execution::ArtifactRequest request;
                request.path = path;
                request.target = target;
                return family_artifact_readiness_dict(
                    self.load_family_artifact(request));
            },
            py::arg("path"),py::arg("target")="host");
    return evaluator;
}


void bind_mace_nonlinear_kokkos(py::module_& module)
{
    module.def(
        "_analyze_mh1_family_architecture",
        [](const std::string& serialized_model) {
            const auto data = nlohmann::json::parse(serialized_model);
            const auto model_type =
                data.value("model_type", std::string());
            MaceBackboneSchema schema;
            try {
                if (model_type == symmetrix::mace_polar::model_type) {
                    schema = MaceBackboneSchema::polar;
                    symmetrix::mace_polar::validate_mace_polar_schema(data);
                } else {
                    schema = MaceBackboneSchema::mh1;
                    validate_mace_nonlinear_schema(data);
                }
            } catch (const std::exception& error) {
                MaceMH1FamilyDescriptor rejected;
                rejected.rejection_reason =
                    model_type == symmetrix::mace_polar::model_type
                    ? std::string("Polar MH-1 identity validation failed: ")
                        + error.what()
                    : std::string("MH-1 identity validation failed: ")
                        + error.what();
                return mh1_family_descriptor_dict(rejected);
            }
            return mh1_family_descriptor_dict(
                analyze_mh1_family_architecture(data, schema));
        },
        py::arg("serialized_model"));
    // The stage description the evaluator orchestrates through, exposed in the
    // contract's schedule vocabulary so its equality with the contract's own
    // generated schedule is a test rather than an assumption.
    module.def(
        "mace_nonlinear_stage_description",
        [](const std::size_t layer_count) {
            return stage_description_dict(
                symmetrix::mace_nonlinear::StageDescription::canonical(
                    layer_count));
        },
        py::arg("layer_count"));
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearKokkos>(
        module,"MACENonlinearKokkos");
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearFloatKokkos>(
        module,"MACENonlinearKokkosFloat");
    auto bind_backbone = [&]<typename Evaluator>(const char* name) {
        return bind_mace_nonlinear_kokkos_evaluator<Evaluator>(module, name);
    };
    symmetrix::mace_polar::bind_mace_polar_kokkos_evaluators(
        module, bind_backbone);
}
