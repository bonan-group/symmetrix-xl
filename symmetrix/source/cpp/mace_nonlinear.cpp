#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mace_nonlinear.hpp"

namespace py = pybind11;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void bind_mace_nonlinear(py::module_& module)
{
    py::class_<MaceNonlinear>(module, "MACENonlinear")
        .def(py::init<const std::string&, const std::string&>(),
            py::arg("filename"), py::arg("head") = "")
        .def_readonly("atomic_numbers", &MaceNonlinear::atomic_numbers)
        .def_readonly("atomic_energies", &MaceNonlinear::atomic_energies)
        .def_readonly("selected_head", &MaceNonlinear::selected_head)
        .def_readonly("available_heads", &MaceNonlinear::available_heads)
        .def_readonly("r_cut", &MaceNonlinear::r_cut)
        .def_readonly("has_field_coupling", &MaceNonlinear::has_field_coupling)
        .def_property_readonly("is_mh1_family", &MaceNonlinear::is_mh1_family)
        .def_property_readonly("mh1_node_channels", &MaceNonlinear::mh1_node_channels)
        .def_property_readonly("mh1_edge_channels", &MaceNonlinear::mh1_edge_channels)
        .def_property_readonly("mh1_radial_size", &MaceNonlinear::mh1_radial_size)
        .def_property_readonly("mh1_l_max", &MaceNonlinear::mh1_l_max)
        .def_property_readonly(
            "mh1_family_rejection_reason",
            &MaceNonlinear::mh1_family_rejection_reason)
        .def_property_readonly(
            "mh1_uses_compiled_products",
            &MaceNonlinear::mh1_uses_compiled_products)
        .def_property_readonly(
            "mh1_uses_pair_conditioning",
            &MaceNonlinear::mh1_uses_pair_conditioning)
        .def_property_readonly(
            "mh1_fast_path_rejection_reason",
            &MaceNonlinear::mh1_fast_path_rejection_reason)
        .def_property_readonly("uses_mh1_fast_path", &MaceNonlinear::uses_mh1_fast_path)
        .def_property_readonly("supports_streamed_edges", &MaceNonlinear::supports_streamed_edges)
        .def_property_readonly("streamed_edges_mode", &MaceNonlinear::streamed_edges_mode)
        .def("set_streamed_edges", &MaceNonlinear::set_streamed_edges)
        .def_property_readonly("edge_workspace_rows", &MaceNonlinear::edge_workspace_rows)
        .def_readonly("node_energies", &MaceNonlinear::node_energies)
        .def_readonly("node_forces", &MaceNonlinear::node_forces)
        .def("compute_node_energies_forces", [](
            MaceNonlinear& self, int num_nodes, IntArray node_types,
            IntArray num_neigh, IntArray neigh_indices, IntArray neigh_types,
            DoubleArray xyz, DoubleArray distances) {
            self.compute_node_energies_forces(
                num_nodes,
                {node_types.data(), static_cast<std::size_t>(node_types.size())},
                {num_neigh.data(), static_cast<std::size_t>(num_neigh.size())},
                {neigh_indices.data(), static_cast<std::size_t>(neigh_indices.size())},
                {neigh_types.data(), static_cast<std::size_t>(neigh_types.size())},
                {xyz.data(), static_cast<std::size_t>(xyz.size())},
                {distances.data(), static_cast<std::size_t>(distances.size())});
        });
}
