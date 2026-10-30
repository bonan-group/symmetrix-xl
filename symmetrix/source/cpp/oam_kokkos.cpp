#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <type_traits>

#include "oam_kokkos.hpp"

namespace py = pybind11;

using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

// Serialization of the neutral family-interface data.  The dict keys are the
// cross-language spelling of the interface.
py::list operand_contracts_to_python(
    const std::vector<symmetrix::execution::BufferContract>& contracts)
{
    py::list result;
    for (const auto& contract : contracts) {
        py::dict entry;
        entry["name"] = contract.name;
        entry["dtype"] = contract.dtype;
        entry["shape"] = contract.shape;
        entry["layout"] = contract.layout.order;
        entry["irreps"] = contract.layout.irreps;
        result.append(entry);
    }
    return result;
}

py::list family_bindings_to_python(
    const std::vector<symmetrix::execution::ImplementationBinding>& bindings)
{
    py::list result;
    for (const auto& binding : bindings) {
        py::dict entry;
        entry["operator_id"] = binding.operator_id;
        entry["kind"] = !binding.custom_kind.empty()
            ? binding.custom_kind
            : operator_kind_name(binding.kind);
        entry["capability_key"] = binding.capability_key;
        entry["implementation_id"] = binding.implementation_id;
        entry["selected_implementation_id"] = binding.selected_implementation_id;
        entry["tier"] = implementation_tier_name(binding.tier);
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
        result.append(entry);
    }
    return result;
}

py::dict artifact_readiness_to_python(
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

// Read one string field from the caller's generated-artifact identity mapping.
// A missing or ``None`` entry leaves the field empty so the native loader can
// fill it from the JIT manifest instead.
std::string identity_field_string(const py::dict& mapping, const char* key)
{
    if (!mapping.contains(key))
        return {};
    const py::object value = mapping[key];
    if (value.is_none())
        return {};
    return py::cast<std::string>(value);
}

// Read one integer extent; a missing or ``None`` entry keeps the native
// sentinel (-1) that means "fill from the manifest or the active device".
std::int32_t identity_field_extent(const py::dict& mapping, const char* key)
{
    if (!mapping.contains(key))
        return -1;
    const py::object value = mapping[key];
    if (value.is_none())
        return -1;
    return py::cast<std::int32_t>(value);
}

// Translate the Python mapping the calculator builds from the artifact manifest
// into the native identity struct. Explicit fields are compared with the
// manifest before missing values are filled, so no caller override wins over
// published artifact identity.
symmetrix::execution::CudaR1ArtifactIdentity parse_artifact_identity(
    const py::object& identity)
{
    symmetrix::execution::CudaR1ArtifactIdentity parsed;
    if (identity.is_none())
        return parsed;
    const py::dict mapping = py::cast<py::dict>(identity);
    parsed.artifact_id = identity_field_string(mapping, "artifact_id");
    parsed.contract_fingerprint =
        identity_field_string(mapping, "contract_fingerprint");
    parsed.semantic_fingerprint =
        identity_field_string(mapping, "semantic_fingerprint");
    parsed.structure_fingerprint =
        identity_field_string(mapping, "structure_fingerprint");
    parsed.target_architecture =
        identity_field_string(mapping, "target_architecture");
    parsed.manifest_path = identity_field_string(mapping, "manifest_path");
    parsed.target_compute_capability =
        identity_field_extent(mapping, "target_compute_capability");
    parsed.max_threads_per_block =
        identity_field_extent(mapping, "max_threads_per_block");
    parsed.target_policy = identity_field_string(mapping, "target_policy");
    parsed.precision_policy = identity_field_string(mapping, "precision_policy");
    parsed.schedule_qualification =
        identity_field_string(mapping, "schedule_qualification");
    return parsed;
}

}  // namespace


template <typename Precision>
void bind_nequip_oam_kokkos_impl(py::module_& module, const char* class_name)
{
    py::class_<NequIPOAMKokkos<Precision>>(module, class_name)
        .def(py::init([] (
                const std::string& filename,
                const std::string& host_plugin_path,
                const std::string& cuda_plugin_path,
                const py::object& cuda_artifact_identity,
                const std::string& host_active_target,
                const std::string& host_artifact_origin,
                const std::string& requested_profile,
                const std::string& requested_edge_variant,
                const std::string& requested_precision) {
            // ``NequIPOAMKokkos`` owns a Kokkos guard and loaded plugins, so
            // it is neither copyable nor movable; the factory returns a raw
            // pointer, which pybind11 adopts exactly like the NequIP binding.
            return new NequIPOAMKokkos<Precision>(
                filename,
                host_plugin_path,
                cuda_plugin_path,
                parse_artifact_identity(cuda_artifact_identity),
                host_active_target,
                host_artifact_origin,
                requested_profile,
                requested_edge_variant,
                requested_precision);
        }),
            py::arg("filename"), py::arg("host_plugin_path") = "",
            py::arg("cuda_plugin_path") = "",
            py::arg("cuda_artifact_identity") = py::none(),
            py::arg("host_active_target") = "",
            py::arg("host_artifact_origin") = "",
            py::arg("requested_profile") = "",
            py::arg("requested_edge_variant") = "",
            py::arg("requested_precision") = "")
        .def_property_readonly("cuda_r1_ready",
            &NequIPOAMKokkos<Precision>::cuda_r1_ready)
        .def_property_readonly("cuda_r1_reason",
            &NequIPOAMKokkos<Precision>::cuda_r1_reason,
            py::return_value_policy::copy)
        .def_property_readonly("cuda_r1_artifact_id",
            &NequIPOAMKokkos<Precision>::cuda_r1_artifact_id)
        .def_property_readonly("cuda_r1_plugin_path",
            &NequIPOAMKokkos<Precision>::cuda_r1_plugin_path)
        .def_property_readonly("cuda_r1_manifest_path",
            &NequIPOAMKokkos<Precision>::cuda_r1_manifest_path)
        .def_property_readonly("cuda_r1_launch_count",
            &NequIPOAMKokkos<Precision>::cuda_r1_launch_count)
        .def_property_readonly("cuda_r1_fallback_count",
            &NequIPOAMKokkos<Precision>::cuda_r1_fallback_count)
        .def_property_readonly("cuda_r1_grid_blocks",
            &NequIPOAMKokkos<Precision>::cuda_r1_grid_blocks)
        .def_property_readonly("atomic_numbers", &NequIPOAMKokkos<Precision>::atomic_numbers)
        .def_property_readonly("r_cut", &NequIPOAMKokkos<Precision>::r_cut)
        .def_property_readonly("num_layers", &NequIPOAMKokkos<Precision>::num_layers)
        .def_property_readonly("last_execution_diagnostic",
            &NequIPOAMKokkos<Precision>::last_execution_diagnostic)
        .def_property_readonly("execution_diagnostic",
            &NequIPOAMKokkos<Precision>::execution_diagnostic)
        .def_property_readonly("execution_backend",
            &NequIPOAMKokkos<Precision>::execution_backend)
        .def_property_readonly("supports_stress", &NequIPOAMKokkos<Precision>::supports_stress)
        .def_property_readonly("supports_streamed_edges",
            &NequIPOAMKokkos<Precision>::supports_streamed_edges)
        .def_property_readonly("streamed_edges_mode",
            &NequIPOAMKokkos<Precision>::streamed_edges_mode)
        .def_property_readonly("execution_workspace_bytes",
            &NequIPOAMKokkos<Precision>::execution_workspace_bytes)
        .def_property_readonly("execution_topology_bytes",
            &NequIPOAMKokkos<Precision>::execution_topology_bytes)
        .def_property_readonly("execution_linear_workspace_bytes",
            &NequIPOAMKokkos<Precision>::execution_linear_workspace_bytes)
        .def_property_readonly("execution_active_edge_count",
            &NequIPOAMKokkos<Precision>::execution_active_edge_count)
        .def_property_readonly("execution_chunk_count",
            &NequIPOAMKokkos<Precision>::execution_chunk_count)
        .def_property_readonly("execution_reduction_mode",
            &NequIPOAMKokkos<Precision>::execution_reduction_mode)
        .def_property_readonly("graph_generation",
            &NequIPOAMKokkos<Precision>::graph_generation)
        .def_property_readonly("geometry_generation",
            &NequIPOAMKokkos<Precision>::geometry_generation)
        .def_property_readonly("jit_oam_host_plugin_ready",
            &NequIPOAMKokkos<Precision>::jit_oam_host_plugin_ready)
        .def_property_readonly("jit_oam_host_plugin_path",
            &NequIPOAMKokkos<Precision>::jit_oam_host_plugin_path)
        .def_property_readonly("jit_oam_host_plugin_artifact_id",
            &NequIPOAMKokkos<Precision>::jit_oam_host_plugin_artifact_id)
        .def_readonly("node_energies", &NequIPOAMKokkos<Precision>::node_energies)
        .def_readonly("node_forces", &NequIPOAMKokkos<Precision>::node_forces)
        .def_readonly("atom_forces", &NequIPOAMKokkos<Precision>::atom_forces)
        .def_readonly("stress_tensor", &NequIPOAMKokkos<Precision>::stress_tensor)
        .def("set_streamed_edges", &NequIPOAMKokkos<Precision>::set_streamed_edges,
            py::arg("mode"))
        .def("_set_execution_plan_request",
            &NequIPOAMKokkos<Precision>::set_execution_plan_request,
            py::arg("algorithm"), py::arg("profile"))
        .def("_set_execution_edge_variant",
            &NequIPOAMKokkos<Precision>::set_execution_edge_variant,
            py::arg("variant"))
        .def_property_readonly("execution_edge_variant",
            &NequIPOAMKokkos<Precision>::execution_edge_variant)
        .def_property_readonly("execution_edge_variant_reason",
            &NequIPOAMKokkos<Precision>::execution_edge_variant_reason)
        .def("_set_execution_plan_qualification",
            &NequIPOAMKokkos<Precision>::set_execution_plan_qualification,
            py::arg("plan_id"), py::arg("artifact_identity"),
            py::arg("backend"), py::arg("precision"), py::arg("schedule"),
            py::arg("capability"), py::arg("empirical_status"),
            py::arg("estimated_bytes"),
            py::arg("available_bytes"), py::arg("reserve_bytes"))
        .def("compute_node_energies_forces", [] (
                NequIPOAMKokkos<Precision>& self, const int num_nodes,
                const IntArray node_types, const IntArray num_neigh,
                const IntArray neigh_indices, const IntArray neigh_types,
                const DoubleArray xyz, const DoubleArray distances) {
            self.compute_node_energies_forces(
                num_nodes, {node_types.data(), static_cast<std::size_t>(node_types.size())},
                {num_neigh.data(), static_cast<std::size_t>(num_neigh.size())},
                {neigh_indices.data(), static_cast<std::size_t>(neigh_indices.size())},
                {neigh_types.data(), static_cast<std::size_t>(neigh_types.size())},
                {xyz.data(), static_cast<std::size_t>(xyz.size())},
                {distances.data(), static_cast<std::size_t>(distances.size())});
        }, py::arg("num_nodes"), py::arg("node_types"), py::arg("num_neigh"),
            py::arg("neigh_indices"), py::arg("neigh_types"), py::arg("xyz"), py::arg("distances"))
        .def("prepare_graph", [] (
                NequIPOAMKokkos<Precision>& self, const int num_nodes,
                const IntArray node_types, const IntArray num_neigh,
                const IntArray neigh_indices, const IntArray neigh_types) {
            return self.prepare_graph(
                num_nodes, {node_types.data(), static_cast<std::size_t>(node_types.size())},
                {num_neigh.data(), static_cast<std::size_t>(num_neigh.size())},
                {neigh_indices.data(), static_cast<std::size_t>(neigh_indices.size())},
                {neigh_types.data(), static_cast<std::size_t>(neigh_types.size())});
        }, py::arg("num_nodes"), py::arg("node_types"), py::arg("num_neigh"),
            py::arg("neigh_indices"), py::arg("neigh_types"))
        .def("prepare_geometry", [] (
                NequIPOAMKokkos<Precision>& self, const std::uint64_t generation,
                const DoubleArray reference_positions, const DoubleArray reference_xyz,
                const DoubleArray cell, const DoubleArray inverse_cell, const IntArray pbc) {
            self.prepare_geometry(
                generation,
                {reference_positions.data(), static_cast<std::size_t>(reference_positions.size())},
                {reference_xyz.data(), static_cast<std::size_t>(reference_xyz.size())},
                {cell.data(), static_cast<std::size_t>(cell.size())},
                {inverse_cell.data(), static_cast<std::size_t>(inverse_cell.size())},
                {pbc.data(), static_cast<std::size_t>(pbc.size())});
        }, py::arg("graph_generation"), py::arg("reference_positions"), py::arg("reference_xyz"),
            py::arg("cell"), py::arg("inverse_cell"), py::arg("pbc"))
        .def("compute_prepared", [] (
                NequIPOAMKokkos<Precision>& self, const std::uint64_t generation,
                const DoubleArray xyz, const DoubleArray distances) {
            self.compute_prepared(
                generation, {xyz.data(), static_cast<std::size_t>(xyz.size())},
                {distances.data(), static_cast<std::size_t>(distances.size())});
        }, py::arg("graph_generation"), py::arg("xyz"), py::arg("distances"))
        .def("reduce_forces", &NequIPOAMKokkos<Precision>::reduce_forces,
            py::arg("graph_generation"))
        .def("reduce_stress", &NequIPOAMKokkos<Precision>::reduce_stress,
            py::arg("volume"), py::arg("graph_generation"))
        .def("compute_stress", &NequIPOAMKokkos<Precision>::compute_stress)
        .def_property_readonly("family_capabilities",
            [] (const NequIPOAMKokkos<Precision>& self) {
                const auto capabilities = self.family_capabilities();
                py::dict result;
                result["target"] = capabilities.target;
                result["precision"] = capabilities.precision;
                result["reference"] = capabilities.reference_ready;
                result["portable"] = capabilities.portable_ready;
                result["generated"] = capabilities.generated_ready;
                result["stress"] = capabilities.stress;
                result["field"] = capabilities.field;
                result["parameter_gradients"] = capabilities.parameter_gradients;
                result["host_artifact"] = capabilities.loads_host_artifact;
                result["device_artifact"] = capabilities.loads_device_artifact;
                return result;
            })
        .def_property_readonly("family_program",
            [] (const NequIPOAMKokkos<Precision>& self) {
                const auto program = self.compile_program();
                py::dict result;
                result["family"] = program.family;
                result["precision"] = program.precision;
                result["version"] = program.version;
                result["fingerprint"] = program.fingerprint;
                py::list operators;
                for (const auto& node : program.operators) {
                    py::dict entry;
                    entry["kind"] = operator_kind_name(node);
                    entry["id"] = node.id;
                    entry["capability_key"] = node.capability_key;
                    py::dict parameters;
                    for (const auto& [name, value] : node.parameters)
                        parameters[py::str(name)] = value;
                    entry["parameters"] = parameters;
                    entry["inputs"] = operand_contracts_to_python(node.inputs);
                    entry["outputs"] = operand_contracts_to_python(node.outputs);
                    operators.append(entry);
                }
                result["operators"] = operators;
                return result;
            })
        .def_property_readonly("family_binding_report",
            [] (const NequIPOAMKokkos<Precision>& self) {
                const auto report = self.family_binding_report();
                py::dict result;
                result["version"] = report.version;
                result["target"] = report.target;
                result["precision"] = report.precision;
                result["artifact_ready"] = report.artifact_ready;
                result["admitted"] = report.admitted;
                result["node_count"] = report.node_count;
                result["portable_node_count"] = report.portable_node_count;
                result["bindings"] = family_bindings_to_python(report.bindings);
                return result;
            })
        .def_property_readonly("execution_plan_report",
            [] (const NequIPOAMKokkos<Precision>& self) {
                const auto& report = self.execution_plan_report();
                py::dict result;
                result["requested_algorithm"] =
                    execution_mode_name(report.requested_algorithm);
                result["requested_profile"] =
                    symmetrix::execution::execution_profile_name(
                        report.requested_profile);
                result["selection_source"] = report.selection_source;
                result["state"] = report.state;
                result["selected_id"] = report.selected_id;
                result["edge_variant"] = self.execution_edge_variant();
                result["edge_variant_reason"] =
                    self.execution_edge_variant_reason();
                const auto origin = self.execution_artifact_origin();
                if (origin.empty())
                    result["artifact_origin"] = py::none();
                else
                    result["artifact_origin"] = origin;
                result["loaded_artifact_identity"] =
                    self.execution_loaded_artifact_identity();
                result["qualification_artifact_identity"] =
                    self.execution_plan_qualification_artifact_identity(
                        report.selected_id);
                result["forward_owner_schedule"] =
                    self.execution_forward_owner_schedule();
                result["reverse_owner_schedule"] =
                    self.execution_reverse_owner_schedule();
                result["tiling_policy"] = self.execution_tiling_policy();
                result["vector_policy"] = self.execution_vector_policy();
                result["shared_memory_policy"] =
                    self.execution_shared_memory_policy();
                result["register_budget_policy"] =
                    self.execution_register_budget_policy();
                result["direct_geometry_policy"] =
                    self.execution_direct_geometry_policy();
                result["channel_tile"] = self.execution_channel_tile();
                result["scalar_width"] = self.execution_scalar_width();
                result["required_isa"] = self.execution_required_isa();
                result["reduction_policy"] = self.execution_reduction_policy();
                result["target_compute_capability"] =
                    self.execution_target_compute_capability();
                result["precision_policy"] = self.execution_precision_policy();
                result["schedule_qualification"] =
                    self.execution_schedule_qualification();
                result["full_message_launch_count"] =
                    self.execution_full_message_launch_count();
                result["launch_count"] =
                    self.execution_full_message_launch_count();
                result["fallback_count"] = self.cuda_r1_fallback_count();
                result["active_bytes"] = self.execution_arena_active_bytes();
                result["selection_reason"] = report.selection_reason;
                if (self.execution_plan_memory_available()) {
                    result["available_bytes"] = report.available_bytes;
                    result["reserve_bytes"] = report.reserve_bytes;
                } else {
                    result["available_bytes"] = py::none();
                    result["reserve_bytes"] = py::none();
                }
                result["boundary_attempt"] = report.boundary_attempt;
                py::list candidates;
                for (const auto& candidate : report.candidates) {
                    py::dict entry;
                    entry["id"] = candidate.id;
                    entry["qualified"] = candidate.qualified;
                    entry["estimated_bytes"] = candidate.estimated_bytes
                        ? py::cast(*candidate.estimated_bytes) : py::none();
                    entry["reason"] = candidate.reason;
                    entry["backend"] = self.family_capabilities().target;
                    entry["precision"] =
                        std::is_same_v<Precision, float> ? "float32" : "float64";
                    entry["schedule"] = candidate.id == "oam-direct-speed"
                        ? "throughput"
                        : candidate.id == "oam-direct-capacity"
                        ? "capacity"
                        : "fixed_workspace";
                    entry["edge_variant"] =
                        candidate.id.rfind("oam-direct-", 0) == 0
                        ? self.execution_edge_variant() : "full_message";
                    entry["artifact_origin"] = candidate.id.rfind(
                        "oam-direct-", 0) == 0
                        ? py::cast(self.execution_artifact_origin())
                        : py::cast(std::string("none"));
                    entry["required_capability"] = candidate.id
                            == "oam-direct-speed"
                        ? "generated_edge_tensor_product"
                        : candidate.id == "oam-direct-capacity"
                        ? "exact_capacity_layout"
                        : "fixed_workspace_layout";
                    entry["empirical_qualification"] =
                        self.execution_plan_empirical_qualification(candidate.id);
                    entry["artifact_identity"] =
                        self.execution_plan_qualification_artifact_identity(
                            candidate.id);
                    entry["estimate_available"] =
                        candidate.estimated_bytes.has_value();
                    entry["fixed_workspace"] = candidate.fixed_workspace;
                    entry["estimate_scope"] = candidate.estimated_bytes
                        ? "exact active transient bytes from the evaluator arena layout"
                        : "unavailable fixed workspace";
                    candidates.append(entry);
                }
                result["candidates"] = candidates;
                // The per-node tiers the plan binds, versioned the same way the
                // other families' reports are.
                result["binding_report_version"] = report.binding_report_version;
                result["bindings"] = family_bindings_to_python(report.bindings);
                return result;
            })
        .def_property_readonly("execution_memory_report",
            [] (const NequIPOAMKokkos<Precision>& self) {
                py::dict result;
                result["active_bytes"] = self.execution_arena_active_bytes();
                result["retained_capacity_bytes"] =
                    self.execution_arena_retained_capacity_bytes();
                result["peak_transient_bytes"] =
                    self.execution_arena_peak_transient_bytes();
                result["replacements"] = self.execution_arena_replacements();
                result["high_water_reason"] =
                    self.execution_arena_high_water_reason();
                result["runtime_subview_count"] =
                    self.execution_arena_runtime_subview_count();
                result["canary_initializations"] =
                    self.execution_arena_canary_initializations();
                result["canaries_intact"] =
                    self.execution_arena_canaries_intact();
                result["arena_bilinear_workspace_bytes"] =
                    self.execution_bilinear_workspace_bytes();
                result["arena_packed_linear_workspace_bytes"] =
                    self.execution_packed_linear_workspace_bytes();
                result["independent_graph_temporary_bytes"] =
                    self.execution_independent_graph_temporary_bytes();
                return result;
            })
        .def_property_readonly("family_artifact_ready",
            &NequIPOAMKokkos<Precision>::family_artifact_ready)
        .def_property_readonly("family_artifact_readiness",
            [] (const NequIPOAMKokkos<Precision>& self) {
                return artifact_readiness_to_python(
                    self.family_artifact_readiness());
            })
        .def_property_readonly("family_diagnostics",
            [] (const NequIPOAMKokkos<Precision>& self) {
                const auto diagnostics = self.family_diagnostics();
                py::dict result;
                result["fallback_count"] = diagnostics.fallback_count;
                result["diagnostic"] = diagnostics.diagnostic;
                return result;
            })
        .def("load_family_artifact",
            [] (NequIPOAMKokkos<Precision>& self, const std::string& path,
                const std::string& target) {
                symmetrix::execution::ArtifactRequest request;
                request.path = path;
                request.target = target;
                return artifact_readiness_to_python(
                    self.load_family_artifact(request));
            },
            py::arg("path"), py::arg("target") = "host");
}

void bind_nequip_oam_kokkos(py::module_& module)
{
    bind_nequip_oam_kokkos_impl<double>(module, "NequIPOAMKokkos");
    bind_nequip_oam_kokkos_impl<float>(module, "NequIPOAMKokkosFloat");
    module.def("_debug_corrupt_oam_execution_arena_guard",
        [] (NequIPOAMKokkos<double>& evaluator) {
            NequIPOAMArenaTestAccess<double>::corrupt_first_guard(evaluator);
        });
    module.def("_debug_corrupt_oam_execution_arena_guard",
        [] (NequIPOAMKokkos<float>& evaluator) {
            NequIPOAMArenaTestAccess<float>::corrupt_first_guard(evaluator);
        });
}
