#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "mace_kokkos_binding_helpers.hpp"
#include "mace_les_kokkos.hpp"
#include "utilities_kokkos.hpp"

namespace py = pybind11;
using ContiguousIntArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using ContiguousDoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;
using symmetrix::python::detail::shaped_array;
using symmetrix::python::detail::view_prefix_1d;
using symmetrix::python::detail::view_prefix_3d;

template <typename Precision>
void bind_mace_les_kokkos_type(py::module_& module, const char* name)
{
    using ContiguousPrecisionArray =
        py::array_t<Precision, py::array::c_style | py::array::forcecast>;

    py::class_<MACELESKokkos<Precision>, MACEKokkos<Precision>>(module, name)
        .def(py::init<const std::string&, const std::string&>(),
             py::arg("filename"), py::arg("head") = "")
        .def_property_readonly("is_mace_les", &MACELESKokkos<Precision>::is_mace_les)
        .def("_configure_les_particle_mesh_for_testing",
            &MACELESKokkos<Precision>::configure_les_particle_mesh_for_testing,
            py::arg("nx"), py::arg("ny"), py::arg("nz"),
            py::arg("spline_order"))
        .def("_use_explicit_les_periodic_for_testing",
            &MACELESKokkos<Precision>::use_explicit_les_periodic_for_testing)
        .def_static("_les_particle_mesh_stream_probe_for_testing", [] () {
            const auto [available, passed, reason, instance_id, default_id,
                particle_count_fences] =
                MACELESKokkos<Precision>::les_particle_mesh_stream_probe_for_testing();
            py::dict result;
            result["available"] = available;
            result["passed"] = passed;
            result["reason"] = reason;
            result["execution_instance_id"] = instance_id;
            result["default_instance_id"] = default_id;
            result["particle_count_fences"] = particle_count_fences;
            return result;
        })
        .def("_validate_les_position_count_for_testing",
            &MACELESKokkos<Precision>::validate_les_position_count,
            py::arg("position_count"))
        .def("_set_les_charge_adjoint_accumulation_for_testing",
            [] (MACELESKokkos<Precision>& self, const bool enabled) {
                self.les_.charge_adjoint_accumulation_enabled = enabled;
            })
        .def("_set_les_geometry",
            [] (MACELESKokkos<Precision>& self,
                    ContiguousDoubleArray positions,
                    ContiguousDoubleArray cell,
                    ContiguousIntArray pbc) {
                const auto positions_info = positions.request();
                const auto cell_info = cell.request();
                const auto pbc_info = pbc.request();
                if (positions_info.ndim != 2 || positions_info.shape[1] != 3
                    || cell_info.ndim != 2 || cell_info.shape[0] != 3
                    || cell_info.shape[1] != 3
                    || pbc_info.ndim != 1 || pbc_info.shape[0] != 3)
                    throw std::invalid_argument(
                        "LES geometry requires positions [N,3], cell [3,3], and PBC [3].");
                self.set_les_geometry(
                    std::span<const double>(
                        positions.data(), positions.size()),
                    std::span<const double>(cell.data(), cell.size()),
                    std::span<const int>(pbc.data(), pbc.size()));
            },
            py::arg("positions"), py::arg("cell"), py::arg("pbc"))
        .def("_evaluate_les_charge_readouts_for_testing",
            [] (MACELESKokkos<Precision>& self,
                ContiguousPrecisionArray h1,
                ContiguousDoubleArray h2) {
                const auto h1_info = h1.request();
                const auto h2_info = h2.request();
                if (h1_info.ndim != 2 || h2_info.ndim != 2
                    || h1_info.shape[0] != h2_info.shape[0]
                    || h1_info.shape[1] != self.num_channels
                    || h2_info.shape[1] != self.num_channels)
                    throw std::invalid_argument(
                        "LES test features must both have shape [nodes, channels].");
                const int nodes = static_cast<int>(h1_info.shape[0]);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H1, nodes, self.num_LM, self.num_channels);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H2, nodes, self.num_channels);
                auto host_h1 = Kokkos::create_mirror_view(self.H1);
                auto host_h2 = Kokkos::create_mirror_view(self.H2);
                Kokkos::deep_copy(host_h1, Precision(0));
                const auto* h1_data = static_cast<const Precision*>(h1_info.ptr);
                const auto* h2_data = static_cast<const double*>(h2_info.ptr);
                for (int node=0; node<nodes; ++node)
                    for (int channel=0; channel<self.num_channels; ++channel) {
                        const auto index = static_cast<std::size_t>(node)
                            *self.num_channels+channel;
                        host_h1(node,0,channel) = h1_data[index];
                        host_h2(node,channel) = h2_data[index];
                    }
                Kokkos::deep_copy(self.H1, host_h1);
                Kokkos::deep_copy(self.H2, host_h2);
                self.compute_les_charges(nodes);
            }, py::arg("h1_scalars"), py::arg("h2_scalars"))
        .def("_prepare_les_charge_readouts_for_testing",
            [] (MACELESKokkos<Precision>& self,
                ContiguousPrecisionArray h1,
                ContiguousDoubleArray h2) {
                const auto h1_info = h1.request();
                const auto h2_info = h2.request();
                if (h1_info.ndim != 2 || h2_info.ndim != 2
                    || h1_info.shape[0] != h2_info.shape[0]
                    || h1_info.shape[1] != self.num_channels
                    || h2_info.shape[1] != self.num_channels)
                    throw std::invalid_argument(
                        "LES test features must both have shape [nodes, channels].");
                const int nodes = static_cast<int>(h1_info.shape[0]);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H1, nodes, self.num_LM, self.num_channels);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H2, nodes, self.num_channels);
                auto host_h1 = Kokkos::create_mirror_view(self.H1);
                auto host_h2 = Kokkos::create_mirror_view(self.H2);
                Kokkos::deep_copy(host_h1, Precision(0));
                const auto* h1_data = static_cast<const Precision*>(h1_info.ptr);
                const auto* h2_data = static_cast<const double*>(h2_info.ptr);
                for (int node=0; node<nodes; ++node)
                    for (int channel=0; channel<self.num_channels; ++channel) {
                        const auto index = static_cast<std::size_t>(node)
                            *self.num_channels+channel;
                        host_h1(node,0,channel) = h1_data[index];
                        host_h2(node,channel) = h2_data[index];
                    }
                Kokkos::deep_copy(self.H1, host_h1);
                Kokkos::deep_copy(self.H2, host_h2);
                self.prepare_les_charge_readouts(nodes);
            }, py::arg("h1_scalars"), py::arg("h2_scalars"))
        .def("_compose_prepared_les_for_testing",
            [] (MACELESKokkos<Precision>& self, const int nodes) {
                self.evaluate_les_term(nodes);
                if (self.use_factorized_async_inference()) {
                    self.enqueue_les_energy_copy();
                    self.factorized_execution_space.fence(
                        "Publish prepared LES test result");
                    self.publish_les_energy();
                }
                py::dict result;
                result["latent_charges"] = shaped_array(
                    view_prefix_1d(self.les_.latent_charges, nodes),
                    {nodes});
                result["les_energy"] = self.les_.energy;
                result["node_energies"] = shaped_array(
                    view_prefix_1d(self.node_energies, nodes),
                    {nodes});
                result["h1_adjoint"] = shaped_array(
                    view_prefix_3d(
                        self.H1_adj, nodes, self.num_LM, self.num_channels),
                    {nodes, self.num_LM, self.num_channels});
                result["h2_adjoint"] = shaped_array(
                    view2vector(self.H2_adj),
                    {nodes, self.num_channels});
                result["explicit_forces"] = shaped_array(
                    view_prefix_1d(
                        self.les_.explicit_forces,
                        3*static_cast<std::size_t>(nodes)),
                    {nodes, 3});
                result["strain_derivative"] = shaped_array(
                    view_prefix_1d(self.les_.strain_derivative, 9),
                    {3, 3});
                return result;
            }, py::arg("num_nodes"))
        .def("_reverse_les_charge_readouts_for_testing",
            [] (MACELESKokkos<Precision>& self,
                ContiguousPrecisionArray h1,
                ContiguousDoubleArray h2,
                ContiguousDoubleArray charge_adjoints,
                py::object h1_seed,
                py::object h2_seed) {
                const auto h1_info = h1.request();
                const auto h2_info = h2.request();
                const auto charge_info = charge_adjoints.request();
                if (h1_info.ndim != 2 || h2_info.ndim != 2
                    || charge_info.ndim != 1
                    || h1_info.shape[0] != h2_info.shape[0]
                    || h1_info.shape[0] != charge_info.shape[0]
                    || h1_info.shape[1] != self.num_channels
                    || h2_info.shape[1] != self.num_channels)
                    throw std::invalid_argument(
                        "LES reverse test inputs have inconsistent shapes.");
                const int nodes = static_cast<int>(h1_info.shape[0]);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H1, nodes, self.num_LM, self.num_channels);
                Kokkos::realloc(
                    Kokkos::WithoutInitializing,
                    self.H2, nodes, self.num_channels);
                auto host_h1 = Kokkos::create_mirror_view(self.H1);
                auto host_h2 = Kokkos::create_mirror_view(self.H2);
                Kokkos::deep_copy(host_h1, Precision(0));
                const auto* h1_data = static_cast<const Precision*>(h1_info.ptr);
                const auto* h2_data = static_cast<const double*>(h2_info.ptr);
                for (int node=0; node<nodes; ++node)
                    for (int channel=0; channel<self.num_channels; ++channel) {
                        const auto index = static_cast<std::size_t>(node)
                            *self.num_channels+channel;
                        host_h1(node,0,channel) = h1_data[index];
                        host_h2(node,channel) = h2_data[index];
                    }
                Kokkos::deep_copy(self.H1, host_h1);
                Kokkos::deep_copy(self.H2, host_h2);

                self.H1_adj = decltype(self.H1_adj)(
                    Kokkos::view_alloc(
                        "LES reverse test H1 adjoint",
                        Kokkos::WithoutInitializing),
                    nodes, self.num_LM, self.num_channels);
                if (self.H2_adj.extent(0) != static_cast<std::size_t>(nodes)
                    || self.H2_adj.extent_int(1) != self.num_channels)
                    Kokkos::realloc(
                        Kokkos::WithoutInitializing,
                        self.H2_adj, nodes, self.num_channels);
                if (h1_seed.is_none()) {
                    Kokkos::deep_copy(self.H1_adj, Precision(0));
                } else {
                    auto seed = h1_seed.cast<ContiguousPrecisionArray>();
                    if (seed.ndim() != 3 || seed.shape(0) != nodes
                        || seed.shape(1) != self.num_LM
                        || seed.shape(2) != self.num_channels)
                        throw std::invalid_argument(
                            "LES H1 test seed has an inconsistent shape.");
                    auto host_seed = Kokkos::create_mirror_view(self.H1_adj);
                    const auto* seed_data = seed.data();
                    std::copy(
                        seed_data, seed_data+self.H1_adj.size(),
                        host_seed.data());
                    Kokkos::deep_copy(self.H1_adj, host_seed);
                }
                if (h2_seed.is_none()) {
                    Kokkos::deep_copy(self.H2_adj, 0.0);
                } else {
                    auto seed = h2_seed.cast<ContiguousDoubleArray>();
                    if (seed.ndim() != 2 || seed.shape(0) != nodes
                        || seed.shape(1) != self.num_channels)
                        throw std::invalid_argument(
                            "LES H2 test seed has an inconsistent shape.");
                    auto host_seed = Kokkos::create_mirror_view(self.H2_adj);
                    std::copy(
                        seed.data(), seed.data()+self.H2_adj.size(),
                        host_seed.data());
                    Kokkos::deep_copy(self.H2_adj, host_seed);
                }
                set_kokkos_view(self.les_.charge_adjoints, charge_adjoints);
                self.compute_les_charges(nodes);
                self.accumulate_les_charge_readout_adjoints(nodes);
                return py::make_tuple(
                    shaped_array(
                        view2vector(self.H1_adj),
                        {nodes, self.num_LM, self.num_channels}),
                    shaped_array(
                        view2vector(self.H2_adj),
                        {nodes, self.num_channels}));
            },
            py::arg("h1_scalars"), py::arg("h2_scalars"),
            py::arg("charge_adjoints"), py::arg("h1_seed") = py::none(),
            py::arg("h2_seed") = py::none())
        .def_property_readonly("les_diagnostics",
            [] (const MACELESKokkos<Precision>& self) {
                py::dict result;
                result["schema"] = self.is_mace_les()
                    ? "symmetrix.model.mace_les" : "";
                result["schema_version"] = self.is_mace_les()
                    ? self.les_.parameters->schema_version : 0;
                if (self.is_mace_les()) {
                    result["kind"] = "monopole";
                    result["sigma"] = self.les_.parameters->sigma;
                    result["dl"] = self.les_.parameters->dl;
                }
                return result;
            })
        .def_property_readonly("_les_execution_space_diagnostics",
            [] (const MACELESKokkos<Precision>& self) {
                const auto selection_name = [] (
                    const bool selected, const bool factorized) {
                    if (!selected)
                        return "unselected";
                    return factorized ? "factorized" : "default";
                };
                py::dict result;
                result["geometry"] = selection_name(
                    self.les_.geometry_execution_space_selected,
                    self.les_.geometry_execution_space_factorized);
                result["evaluation"] = selection_name(
                    self.les_.evaluation_execution_space_selected,
                    self.les_.evaluation_execution_space_factorized);
                return result;
            })
        .def_property_readonly("les_charge_readout_1",
            [] (const MACELESKokkos<Precision>& self) {
                return shaped_array(
                    view_prefix_1d(
                        self.les_.charge_readout_1,
                        self.les_.charge_active_nodes),
                    {self.les_.charge_active_nodes});
            })
        .def_property_readonly("les_charge_readout_2",
            [] (const MACELESKokkos<Precision>& self) {
                return shaped_array(
                    view_prefix_1d(
                        self.les_.charge_readout_2,
                        self.les_.charge_active_nodes),
                    {self.les_.charge_active_nodes});
            })
        .def_property_readonly("latent_charges",
            [] (const MACELESKokkos<Precision>& self) {
                return shaped_array(
                    view_prefix_1d(
                        self.les_.latent_charges, self.les_.charge_active_nodes),
                    {self.les_.charge_active_nodes});
            })
        .def_property_readonly("_les_charge_adjoints_for_testing",
            [] (const MACELESKokkos<Precision>& self) {
                if (!self.les_.results_ready)
                    throw std::logic_error("LES results are not ready.");
                return shaped_array(
                    view_prefix_1d(
                        self.les_.charge_adjoints,
                        self.les_.charge_active_nodes),
                    {self.les_.charge_active_nodes});
            })
        .def_property_readonly("les_energy",
            [] (const MACELESKokkos<Precision>& self) {
                if (!self.les_.results_ready)
                    throw std::logic_error("LES results are not ready.");
                return self.les_.energy;
            })
        .def_property_readonly("les_atomic_energies",
            [] (const MACELESKokkos<Precision>& self) {
                if (!self.les_.results_ready)
                    throw std::logic_error("LES results are not ready.");
                return shaped_array(
                    view_prefix_1d(
                        self.les_.atomic_energies,
                        self.les_.charge_active_nodes),
                    {self.les_.charge_active_nodes});
            })
        .def_property_readonly("les_charge_workspace_diagnostics",
            [] (const MACELESKokkos<Precision>& self) {
                py::dict result;
                result["active_nodes"] = self.les_.charge_active_nodes;
                result["capacity_nodes"] = self.les_.latent_charges.extent(0);
                result["active_bytes"] =
                    self.les_charge_workspace_active_bytes();
                result["capacity_bytes"] =
                    self.les_charge_workspace_capacity_bytes();
                result["replacement_count"] =
                    self.les_.charge_workspace_replacements;
                result["reuse_count"] = self.les_.charge_workspace_reuses;
                return result;
            })
        .def_property_readonly("les_monopole_diagnostics",
            [] (const MACELESKokkos<Precision>& self) {
                if (!self.les_.monopole)
                    throw std::logic_error("LES monopole is not configured.");
                const auto diagnostics = self.les_.monopole->diagnostics();
                py::dict result;
                result["reciprocal_candidate_count"] =
                    diagnostics.reciprocal_candidate_count;
                result["reciprocal_vector_count"] =
                    diagnostics.reciprocal_vector_count;
                result["logical_extent_bytes"] = diagnostics.logical_extent_bytes;
                result["allocated_capacity_bytes"] =
                    diagnostics.allocated_capacity_bytes;
                result["active_index_capacity_bytes"] =
                    diagnostics.active_index_capacity_bytes;
                result["index_pack_count"] = diagnostics.index_pack_count;
                result["table_rebuilds"] = diagnostics.table_rebuilds;
                result["geometry_updates"] = diagnostics.geometry_updates;
                result["active_bytes"] = diagnostics.active_bytes;
                result["capacity_bytes"] = diagnostics.capacity_bytes;
                return result;
            })
        .def_property_readonly("_les_particle_mesh_diagnostics_for_testing",
            [] (const MACELESKokkos<Precision>& self) {
                if (!self.les_.monopole)
                    throw std::logic_error("LES monopole is not configured.");
                const auto diagnostics = self.les_.monopole->diagnostics();
                py::dict result;
                result["periodic_algorithm"] =
                    diagnostics.periodic_algorithm
                        == symmetrix::les::LESPeriodicAlgorithm::particle_mesh
                    ? "particle_mesh" : "explicit_sum";
                result["owner_generation"] =
                    self.les_.particle_mesh_owner_generation;
                result["owner_fences"] = self.les_.particle_mesh_owner_fences;
                result["allocation_generation"] =
                    diagnostics.particle_mesh_allocation_generation;
                result["reservations"] = diagnostics.particle_mesh_reservations;
                result["allocations"] = diagnostics.particle_mesh_allocations;
                result["forward_plan_builds"] =
                    diagnostics.particle_mesh_forward_plan_builds;
                result["forward_plan_reuses"] =
                    diagnostics.particle_mesh_forward_plan_reuses;
                result["inverse_plan_builds"] =
                    diagnostics.particle_mesh_inverse_plan_builds;
                result["inverse_plan_reuses"] =
                    diagnostics.particle_mesh_inverse_plan_reuses;
                result["reciprocal_preparations"] =
                    diagnostics.particle_mesh_reciprocal_preparations;
                result["stencil_preparations"] =
                    diagnostics.particle_mesh_stencil_preparations;
                result["particle_count_fences"] =
                    diagnostics.particle_mesh_particle_count_fences;
                result["execution_instance_id"] =
                    diagnostics.particle_mesh_execution_instance_id;
                result["table_rebuilds"] = diagnostics.table_rebuilds;
                return result;
            })
        ;
}

void bind_mace_les_kokkos(py::module_& module)
{
    bind_mace_les_kokkos_type<double>(module, "MACELESKokkos");
    bind_mace_les_kokkos_type<float>(module, "MACELESKokkosFloat");
}
