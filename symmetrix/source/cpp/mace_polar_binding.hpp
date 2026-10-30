#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "long_range_operator_kinds.hpp"
#include "mace_polar_delta_binding.hpp"
#include "mace_polar_kokkos.hpp"
#include "mace_polar_consumer.hpp"

// The PolarMACE bindings' shared contract-data entry point: the long-range
// slice of a POLAR_BACKBONE contract, exactly as the family's two binding
// sites accept it (the evaluator's configuration setter and the program-slice
// query).  One spelling of the keyword surface keeps the two from drifting.
namespace symmetrix::execution::polar {

polar::PolarLongRangeContract long_range_contract_from_python(
    double kspace_cutoff,
    int density_max_l,
    double density_smearing_width,
    int feature_max_l,
    const std::vector<double>& feature_smearing_widths,
    std::size_t charges_dimension,
    std::size_t field_dimension,
    std::size_t potential_dimension,
    double field_norm_factor,
    int recursion_step_count,
    const std::string& table_dtype,
    bool include_electrostatic_self_interaction = false,
    bool field_si = false);

} // namespace symmetrix::execution::polar

namespace symmetrix::mace_polar {

namespace py = pybind11;

inline py::dict polar_consumer_diagnostics_to_python(
    const ConsumerDiagnostics& d)
{
    py::dict diagnostics;
    diagnostics["requested_solver"]=std::string(
        execution::long_range::to_string(d.evaluation.requested_solver));
    diagnostics["resolved_solver"]=std::string(
        execution::long_range::to_string(d.evaluation.resolved_solver));
    diagnostics["solver_reason"]=d.evaluation.solver_reason;
    diagnostics["automatic_atom_count"]=d.evaluation.automatic_atom_count;
    diagnostics["automatic_reciprocal_mode_count"]=
        d.evaluation.automatic_reciprocal_mode_count;
    diagnostics["automatic_reciprocal_mode_count_is_lower_bound"]=
        d.evaluation.automatic_reciprocal_mode_count_is_lower_bound;
    diagnostics["automatic_atom_mode_pair_count"]=
        d.evaluation.automatic_atom_mode_pair_count;
    diagnostics["automatic_atom_mode_pair_threshold"]=
        d.evaluation.automatic_atom_mode_pair_threshold;
    diagnostics["fallback_count"]=d.evaluation.fallback_count;
    diagnostics["backbone_fallback_count"]=d.evaluation.backbone_fallback_count;
    diagnostics["polar_fallback_count"]=d.evaluation.polar_fallback_count;
    diagnostics["reverse_strategy"]=d.evaluation.reverse_strategy;
    diagnostics["transaction_count"]=d.transaction_count;
    diagnostics["reverse_count"]=d.reverse_count;
    diagnostics["graph_generation"]=d.graph_generation;
    diagnostics["graph_rebuild_count"]=d.graph_rebuild_count;
    diagnostics["graph_reuse_count"]=d.graph_reuse_count;
    diagnostics["topology_validation_count"]=d.topology_validation_count;
    diagnostics["topology_validation_edge_count"]=
        d.topology_validation_edge_count;
    diagnostics["topology_reuse_scan_count"]=d.topology_reuse_scan_count;
    diagnostics["topology_facade_scan_count"]=d.topology_facade_scan_count;
    diagnostics["geometry_update_count"]=d.geometry_update_count;
    diagnostics["geometry_publish_count"]=d.geometry_publish_count;
    diagnostics["geometry_reuse_count"]=d.geometry_reuse_count;
    diagnostics["geometry_allocation_count"]=d.geometry_allocation_count;
    diagnostics["geometry_allocation_bytes"]=d.geometry_allocation_bytes;
    diagnostics["public_output_allocation_count"]=
        d.public_output_allocation_count;
    diagnostics["active_nodes"]=d.active_nodes;
    diagnostics["active_edges"]=d.active_edges;
    diagnostics["input_d2h_bytes"]=d.input_d2h_bytes;
    diagnostics["input_h2d_bytes"]=d.input_h2d_bytes;
    diagnostics["topology_input_d2h_bytes"]=d.topology_input_d2h_bytes;
    diagnostics["metadata_d2h_bytes"]=d.metadata_d2h_bytes;
    diagnostics["public_output_d2h_bytes"]=d.public_output_d2h_bytes;
    diagnostics["scalar_output_d2h_bytes"]=d.scalar_output_d2h_bytes;
    diagnostics["virial_output_d2h_bytes"]=d.virial_output_d2h_bytes;
    diagnostics["force_output_d2h_bytes"]=d.force_output_d2h_bytes;
    diagnostics["output_h2d_bytes"]=d.output_h2d_bytes;
    diagnostics["retained_output_elements"]=d.retained_output_elements;
    diagnostics["reciprocal_allocation_elements"]=
        d.evaluation.reciprocal_allocation_elements;
    diagnostics["reciprocal_allocation_bytes"]=
        d.evaluation.reciprocal_allocation_bytes;
    diagnostics["reciprocal_table_elements"]=
        d.evaluation.reciprocal_table_elements;
    diagnostics["particle_mesh_active_bytes"]=
        d.evaluation.particle_mesh_active_bytes;
    diagnostics["particle_mesh_capacity_bytes"]=
        d.evaluation.particle_mesh_capacity_bytes;
    diagnostics["particle_mesh_volume"]=d.evaluation.particle_mesh_volume;
    diagnostics["particle_mesh_shape"]=d.evaluation.particle_mesh_shape;
    diagnostics["particle_mesh_rebuilds"]=d.evaluation.particle_mesh_rebuilds;
    diagnostics["particle_mesh_reuses"]=d.evaluation.particle_mesh_reuses;
    diagnostics["particle_mesh_forward_plan_builds"]=
        d.evaluation.particle_mesh_forward_plan_builds;
    diagnostics["particle_mesh_forward_plan_reuses"]=
        d.evaluation.particle_mesh_forward_plan_reuses;
    diagnostics["particle_mesh_inverse_plan_builds"]=
        d.evaluation.particle_mesh_inverse_plan_builds;
    diagnostics["particle_mesh_inverse_plan_reuses"]=
        d.evaluation.particle_mesh_inverse_plan_reuses;
    diagnostics["realspace_retained_elements"]=
        d.evaluation.realspace_retained_elements;
    diagnostics["realspace_retained_bytes"]=
        d.evaluation.realspace_retained_bytes;
    diagnostics["diagnostic_materialization_count"]=
        d.evaluation.diagnostic_materialization_count;
    diagnostics["public_output_elements"]=d.evaluation.public_output_elements;
    diagnostics["backbone_evaluation_count"]=
        d.evaluation.backbone_evaluation_count;
    diagnostics["polar_reverse_count"]=d.evaluation.polar_reverse_count;
    return diagnostics;
}

// The polar family's per-graph scalar channel: the program inputs the
// family's equilibration, external-field, and readout nodes consume.  The
// bindings are the family's own -- an MH-1 model has no graph-level scalar
// channel -- so they are declared on the Polar classes only.  The Polar
// family state owns their validation and retention; the shared backbone only
// provides the protected orchestration seam used by that sibling evaluator.
template<typename Evaluator>
void bind_polar_graph_inputs(py::class_<Evaluator>& evaluator)
{
    evaluator
        .def("set_polar_graph_inputs",
            [] (Evaluator& self,const double total_charge,
                const double total_spin,const double fermi_level,
                const std::vector<double>& electric_field) {
                self.set_polar_graph_inputs(
                    total_charge,total_spin,fermi_level,electric_field);
            },
            py::arg("total_charge"),py::arg("total_spin"),
            py::arg("fermi_level"),py::arg("electric_field"),
            "Set the graph-level scalar program inputs of the PolarMACE "
            "family.  The channel's mapping onto the compiled program's "
            "declared buffers is the family's declaration:\n"
            "  * total_charge and total_spin are the per-graph targets the "
            "fukui_normalize node consumes -- the closed-form equilibration "
            "targets (Q + S - 1) / 2 and (Q - S + 1) / 2;\n"
            "  * electric_field (three components) fills lanes 1..3 of the "
            "external_field_projection node's external_potential input, the "
            "displaced-GTO external-field coupling;\n"
            "  * fermi_level is lane 0 of that same buffer, which the frozen "
            "oracle's forward fills with zeros_like(fermi_level) "
            "(mace/modules/extensions.py:643-645) and never couples -- the "
            "recorded reference values are invariant under it.\n"
            "The global nodes that consume these lanes are declared program "
            "data; this channel is validated and retained here for them.")
        .def_property_readonly(
            "polar_graph_inputs",
            [] (const Evaluator& self) {
                py::dict result;
                result["set"] = self.polar_graph_inputs_set();
                result["total_charge"] = self.polar_total_charge();
                result["total_spin"] = self.polar_total_spin();
                result["fermi_level"] = self.polar_fermi_level();
                const std::array<double,3> field = self.polar_electric_field();
                result["electric_field"] =
                    std::vector<double>{field[0],field[1],field[2]};
                return result;
            },
            "The polar graph-input channel this evaluator retains: whether a "
            "frontend has supplied it, and the values it holds.");
}

template<typename Evaluator>
void bind_polar_long_range_contract(py::class_<Evaluator>& evaluator)
{
    evaluator
        .def_property_readonly(
            "polar_long_range_contract",
            [] (const Evaluator& self) {
                const auto& contract=self.polar_long_range_contract();
                if(!contract.has_value())
                    return py::dict();
                const auto& value=*contract;
                py::dict result;
                result["kspace_cutoff"]=value.kspace_cutoff;
                result["density_max_l"]=value.density_max_l;
                result["density_smearing_width"]=value.density_smearing_width;
                result["feature_max_l"]=value.feature_max_l;
                result["feature_smearing_widths"]=value.feature_smearing_widths;
                result["charges_dimension"]=value.charges_dimension;
                result["field_dimension"]=value.field_dimension;
                result["potential_dimension"]=value.potential_dimension;
                result["field_norm_factor"]=value.field_norm_factor;
                result["recursion_step_count"]=value.recursion_step_count;
                result["spin_channel_count"]=value.spin_channel_count;
                result["table_dtype"]=value.table_dtype;
                result["accumulator_dtype"]=value.accumulator_dtype;
                result["include_electrostatic_self_interaction"]=
                    value.include_electrostatic_self_interaction;
                result["field_si"]=value.field_si;
                return result;
            },
            "The Polar family's configured long-range contract.")
        .def("_set_polar_long_range_contract",
            [] (Evaluator& self,const double kspace_cutoff,
                    const int density_max_l,const double density_smearing_width,
                    const int feature_max_l,
                    const std::vector<double>& feature_smearing_widths,
                    const std::size_t charges_dimension,
                    const std::size_t field_dimension,
                    const std::size_t potential_dimension,
                    const double field_norm_factor,
                    const int recursion_step_count,
                    const std::string& table_dtype,
                    const bool include_electrostatic_self_interaction,
                    const bool field_si) {
                self.set_polar_long_range_contract(
                    symmetrix::execution::polar::long_range_contract_from_python(
                        kspace_cutoff,density_max_l,density_smearing_width,
                        feature_max_l,feature_smearing_widths,charges_dimension,
                        field_dimension,potential_dimension,field_norm_factor,
                        recursion_step_count,table_dtype,
                        include_electrostatic_self_interaction,field_si));
            },
            py::arg("kspace_cutoff"),py::arg("density_max_l"),
            py::arg("density_smearing_width"),py::arg("feature_max_l"),
            py::arg("feature_smearing_widths"),py::arg("charges_dimension"),
            py::arg("field_dimension"),py::arg("potential_dimension"),
            py::arg("field_norm_factor")=1.0,
            py::arg("recursion_step_count"),py::arg("table_dtype")="float64",
            py::arg("include_electrostatic_self_interaction")=false,
            py::arg("field_si")=false)
        .def_property_readonly(
            "has_polar_long_range_contract",
            [] (const Evaluator& self) {
                return self.has_polar_long_range_contract();
            })
        // The chain the configured contract declares.  An empty entry list
        // means this evaluator carries no long-range contract -- ask
        // ``has_polar_long_range_contract`` rather than reading the emptiness,
        // which the free ``polar_schedule`` also uses for an invalid depth.
        .def_property_readonly(
            "polar_schedule",
            [] (const Evaluator& self) {
                const auto schedule = self.polar_schedule();
                py::dict result;
                result["version"] = schedule.version;
                result["recursion_step_count"] = schedule.recursion_step_count;
                py::list entries;
                for (const auto& entry : schedule.entries) {
                    py::dict item;
                    item["kind"] =
                        entry.kind
                            == symmetrix::execution::polar::
                                PolarScheduleEntry::Kind::backbone
                        ? "backbone" : "global_operator";
                    item["name"] = entry.name;
                    item["iteration"] = entry.iteration;
                    entries.append(item);
                }
                result["entries"] = entries;
                return result;
            });
}

template<typename BackboneBinder>
void bind_mace_polar_kokkos_evaluators(
    py::module_& module, BackboneBinder&& bind_backbone)
{
    auto evaluator = bind_backbone.template operator()<MacePolarKokkos>(
        "MACEPolarKokkos");
    bind_polar_long_range_contract(evaluator);
    bind_polar_graph_inputs(evaluator);
    bind_polar_delta_surface(evaluator);

    auto float_evaluator =
        bind_backbone.template operator()<MacePolarFloatKokkos>(
            "MACEPolarKokkosFloat");
    bind_polar_long_range_contract(float_evaluator);
    bind_polar_graph_inputs(float_evaluator);
    bind_polar_delta_surface(float_evaluator);

    const auto bind_consumer=[&]<typename Consumer>(const char* name) {
        using DoubleArray=py::array_t<
            double,py::array::c_style|py::array::forcecast>;
        using IntArray=py::array_t<
            int,py::array::c_style|py::array::forcecast>;
        py::class_<Consumer>(module,name)
            .def(py::init<const std::string&,const std::string&>(),
                py::arg("filename"),py::arg("requested_head")="")
            .def_property("streamed_edges",
                [] (const Consumer& self) {
                    return self.streamed_edges_mode();
                },
                [] (Consumer& self,const std::string& mode) {
                    self.set_streamed_edges(mode);
                })
            .def_property_readonly("cutoff",&Consumer::cutoff)
            .def_property_readonly("diagnostics",
                [] (const Consumer& self) {
                    return polar_consumer_diagnostics_to_python(
                        self.diagnostics());
                })
            .def("_arm_graph_geometry_failure_for_testing",
                &Consumer::arm_graph_geometry_failure_for_testing)
            .def("_evaluate_prepared_for_testing",
                [] (Consumer& self,const IntArray& atomic_numbers,
                    const DoubleArray& positions,const DoubleArray& cell,
                    const IntArray& pbc,const IntArray& num_neigh,
                    const IntArray& sources,const IntArray& source_types,
                    const DoubleArray& reference_xyz,const bool replace_graph,
                    const DoubleArray& electric_field,const std::string& solver) {
                    if(atomic_numbers.ndim()!=1
                        ||positions.size()!=3*atomic_numbers.size()
                        ||cell.size()!=9||pbc.size()!=3
                        ||electric_field.size()!=3)
                        throw std::invalid_argument(
                            "MACE Polar prepared test array extents are "
                            "inconsistent.");
                    ConsumerSystemRequest request;
                    request.atomic_numbers={atomic_numbers.data(),
                        static_cast<std::size_t>(atomic_numbers.size())};
                    request.positions={positions.data(),
                        static_cast<std::size_t>(positions.size())};
                    request.cell={cell.data(),static_cast<std::size_t>(cell.size())};
                    request.pbc={pbc.data(),static_cast<std::size_t>(pbc.size())};
                    std::copy(electric_field.data(),electric_field.data()+3,
                        request.electric_field.begin());
                    request.solver_request=execution::long_range::
                        parse_long_range_solver_request(solver);
                    ConsumerPreparedGraphRequest graph;
                    graph.num_neigh={num_neigh.data(),
                        static_cast<std::size_t>(num_neigh.size())};
                    graph.sources={sources.data(),
                        static_cast<std::size_t>(sources.size())};
                    graph.source_types={source_types.data(),
                        static_cast<std::size_t>(source_types.size())};
                    graph.reference_xyz={reference_xyz.data(),
                        static_cast<std::size_t>(reference_xyz.size())};
                    return polar_consumer_diagnostics_to_python(
                        self.evaluate_prepared_device(
                            request,graph,replace_graph).diagnostics);
                },
                py::arg("atomic_numbers"),py::arg("positions"),
                py::arg("cell"),py::arg("pbc"),py::arg("num_neigh"),
                py::arg("sources"),py::arg("source_types"),
                py::arg("reference_xyz"),py::arg("replace_graph"),
                py::arg("electric_field"),py::arg("solver")="auto")
            .def("load_artifact",&Consumer::load_artifact,
                py::arg("path"),py::arg("target")="host")
            .def("load_product_artifact",&Consumer::load_product_artifact,
                py::arg("layer"),py::arg("kind"),py::arg("path"),
                py::arg("metadata_json"),py::arg("schedule"),
                py::arg("persistent_blocks_per_compute_unit"),
                py::arg("backend"))
            .def_property_readonly(
                "product_artifacts_ready",&Consumer::product_artifacts_ready)
            .def("evaluate_system",
                [] (Consumer& self,const IntArray& atomic_numbers,
                    const DoubleArray& positions,const DoubleArray& cell,
                    const IntArray& pbc,const double total_charge,
                    const double total_spin,const double fermi_level,
                    const DoubleArray& electric_field,
                    const std::string& solver,const bool include_forces,
                    const bool include_stress,const bool include_dipole) {
                    if(atomic_numbers.ndim()!=1||positions.size()!=3*atomic_numbers.size()
                        ||cell.size()!=9||pbc.size()!=3||electric_field.size()!=3)
                        throw std::invalid_argument(
                            "MACE Polar consumer array extents are inconsistent.");
                    ConsumerSystemRequest request;
                    request.atomic_numbers={atomic_numbers.data(),
                        static_cast<std::size_t>(atomic_numbers.size())};
                    request.positions={positions.data(),
                        static_cast<std::size_t>(positions.size())};
                    request.cell={cell.data(),static_cast<std::size_t>(cell.size())};
                    request.pbc={pbc.data(),static_cast<std::size_t>(pbc.size())};
                    request.total_charge=total_charge;
                    request.total_spin=total_spin;
                    request.fermi_level=fermi_level;
                    std::copy(electric_field.data(),electric_field.data()+3,
                        request.electric_field.begin());
                    request.solver_request=execution::long_range::
                        parse_long_range_solver_request(solver);
                    request.include_forces=include_forces;
                    request.include_stress=include_stress;
                    request.include_dipole=include_dipole;
                    const auto result=self.evaluate(request);
                    py::dict output;
                    output["energy"]=result.energy;
                    output["free_energy"]=result.free_energy;
                    if(include_forces) {
                        py::array_t<double> values(py::array::ShapeContainer{
                            static_cast<py::ssize_t>(atomic_numbers.size()),
                            py::ssize_t(3)});
                        std::copy(result.forces.begin(),result.forces.end(),
                            values.mutable_data());
                        output["forces"]=std::move(values);
                    }
                    if(include_stress) {
                        py::array_t<double> values(result.stress.size());
                        std::copy(result.stress.begin(),result.stress.end(),
                            values.mutable_data());
                        output["stress"]=std::move(values);
                    }
                    if(include_dipole) {
                        py::array_t<double> values(result.dipole.size());
                        std::copy(result.dipole.begin(),result.dipole.end(),
                            values.mutable_data());
                        output["dipole"]=std::move(values);
                    }
                    output["diagnostics"]=
                        polar_consumer_diagnostics_to_python(result.diagnostics);
                    return output;
                },
                py::arg("atomic_numbers"),py::arg("positions"),
                py::arg("cell"),py::arg("pbc"),py::arg("total_charge")=0.0,
                py::arg("total_spin")=0.0,py::arg("fermi_level")=0.0,
                py::arg("electric_field"),py::arg("solver")="auto",
                py::arg("include_forces")=false,
                py::arg("include_stress")=false,
                py::arg("include_dipole")=false);
    };
    bind_consumer.template operator()<MacePolarConsumer>("MACEPolarConsumer");
    bind_consumer.template operator()<MacePolarConsumerFloat>(
        "MACEPolarConsumerFloat");

    module.def("_polar_consumer_scientific_external_potential",
        [] (const double fermi_level,
            const py::array_t<double,py::array::c_style|py::array::forcecast>&
                electric_field) {
            if(electric_field.size()!=3)
                throw std::invalid_argument(
                    "MACE Polar electric field requires three components.");
            ConsumerSystemRequest request;
            request.fermi_level=fermi_level;
            std::copy(electric_field.data(),electric_field.data()+3,
                request.electric_field.begin());
            const auto potential=
                consumer_scientific_external_potential(request);
            return std::vector<double>(potential.begin(),potential.end());
        },py::arg("fermi_level"),py::arg("electric_field"));

    module.def("_polar_consumer_device_mapping_probe",[] {
        Kokkos::View<double*> public_outputs("polar mapping public",9);
        Kokkos::View<int*> node_indices("polar mapping nodes",2);
        Kokkos::View<double**,Kokkos::LayoutRight> forces(
            "polar mapping forces",3,3);
        Kokkos::View<double*> stress("polar mapping stress",6);
        Kokkos::View<double*> virial("polar mapping virial",6);
        Kokkos::View<double*> isolated("polar mapping isolated virial",6);
        const std::array<double,9> output_values{
            0.0,38.0,49.0,60.0,11.0,22.0,33.0,0.0,0.0};
        const std::array<int,2> mapped{1,0};
        const std::array<double,9> initial{
            3.0,3.0,3.0,3.0,3.0,3.0,70.0,80.0,90.0};
        const std::array<double,6> stress_values{1,2,3,4,5,6};
        Kokkos::deep_copy(public_outputs,Kokkos::View<const double*,
            Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                output_values.data(),output_values.size()));
        Kokkos::deep_copy(node_indices,Kokkos::View<const int*,
            Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(mapped.data(),mapped.size()));
        Kokkos::deep_copy(forces,Kokkos::View<const double**,
            Kokkos::LayoutRight,Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                initial.data(),3,3));
        Kokkos::deep_copy(stress,Kokkos::View<const double*,
            Kokkos::HostSpace,Kokkos::MemoryUnmanaged>(
                stress_values.data(),stress_values.size()));
        add_polar_public_forces_device(
            public_outputs,std::size_t{1},node_indices,forces,std::size_t{2});
        map_polar_stress_to_virial_device(stress,10.0,true,virial);
        map_polar_stress_to_virial_device(stress,10.0,false,isolated);
        Kokkos::fence();
        const auto host_forces=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),forces);
        const auto host_virial=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),virial);
        const auto host_isolated=Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace(),isolated);
        py::dict result;
        py::array_t<double> force_values(
            py::array::ShapeContainer{py::ssize_t(3),py::ssize_t(3)});
        std::copy(host_forces.data(),host_forces.data()+9,
            force_values.mutable_data());
        result["forces"]=std::move(force_values);
        result["virial"]=std::vector<double>(
            host_virial.data(),host_virial.data()+6);
        result["isolated_virial"]=std::vector<double>(
            host_isolated.data(),host_isolated.data()+6);
        return result;
    });
}

} // namespace symmetrix::mace_polar
