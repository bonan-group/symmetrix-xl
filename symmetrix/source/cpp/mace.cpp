#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>

#include "mace.hpp"

namespace py = pybind11;
using ContiguousIntArray =
    py::array_t<int, py::array::c_style | py::array::forcecast>;
using ContiguousDoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

namespace {

template <typename Precision>
std::unique_ptr<MACECPU<Precision>> load_mace_cpu(
    const std::string& filename, const std::string& requested_head)
{
    auto evaluator = std::make_unique<MACECPU<Precision>>(filename, requested_head);
    if (!evaluator->supports_streamed_edges()) {
        if (PyErr_WarnEx(
                PyExc_UserWarning,
                "Loaded the original Symmetrix pair-spline format (named v1 here); using "
                "streamed_edges='materialized'. Re-export with radial_format='compact' "
                "to enable streamed_edges='generic' execution.",
                1) < 0)
            throw py::error_already_set();
    }
    return evaluator;
}

template <typename Precision>
void prepare_active_types(
    MACECPU<Precision>& self,
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
    self.prepare_active_types(types);
}

}  // namespace

template <typename Precision>
void bind_mace_cpu(py::module_ &m, const char* class_name)
{
    using Evaluator = MACECPU<Precision>;
    py::class_<Evaluator>(m, class_name)
        .def(py::init(static_cast<std::unique_ptr<Evaluator> (*)(
                const std::string&, const std::string&)>(load_mace_cpu<Precision>)),
            py::arg("filename"), py::arg("head") = "")
        .def("set_streamed_edges", &Evaluator::set_streamed_edges)
        .def_property_readonly("streamed_edges_mode", &Evaluator::streamed_edges_mode)
        .def_property_readonly("supports_streamed_edges", &Evaluator::supports_streamed_edges)
        .def_property_readonly("scalar_size_bytes", [] (const Evaluator&) {
            return sizeof(Precision);
        })
        .def_property_readonly("R0_storage_size", [] (const Evaluator& self) {
            return self.R0.size()+self.R0_deriv.size();
        })
        .def_property_readonly("R1_storage_size", [] (const Evaluator& self) {
            return self.R1.size()+self.R1_deriv.size();
        })
        .def_readonly("atomic_numbers", &Evaluator::atomic_numbers)
        .def_readonly("atomic_energies", &Evaluator::atomic_energies)
        .def_readonly("selected_head", &Evaluator::selected_head)
        .def_readonly("available_heads", &Evaluator::available_heads)
        .def_readonly("active_atomic_numbers", &Evaluator::active_atomic_numbers)
        .def("prepare_active_types",
            [] (Evaluator& self, ContiguousIntArray node_types) {
                self.prepare_active_types(
                    std::span<const int>(node_types.data(), node_types.size()));
            })
        .def_readonly("r_cut", &Evaluator::r_cut)
        .def_readonly("L_max", &Evaluator::L_max)
        .def_readonly("single_layer_readout", &Evaluator::single_layer_readout)
        .def_readwrite("node_forces", &Evaluator::node_forces)
        .def_readwrite("node_energies", &Evaluator::node_energies)
        .def_readwrite("H0_weights", &Evaluator::H0_weights)
        .def_readwrite("R0", &Evaluator::R0)
        .def_readonly("R0_deriv", &Evaluator::R0_deriv)
        .def_readwrite("R1", &Evaluator::R1)
        .def_readonly("R1_deriv", &Evaluator::R1_deriv)
        .def_readwrite("A0", &Evaluator::A0)
        .def_readwrite("A0_adj", &Evaluator::A0_adj)
        .def_readwrite("M0", &Evaluator::M0)
        .def_readwrite("M0_adj", &Evaluator::M0_adj)
        .def_readwrite("H1", &Evaluator::H1)
        .def_readwrite("H1_adj", &Evaluator::H1_adj)
        .def_readonly(
            "first_interaction_residual",
            &Evaluator::first_interaction_residual)
        .def_readonly(
            "H1_first_residual_weights",
            &Evaluator::H1_first_residual_weights)
        .def_readwrite("has_field_coupling", &Evaluator::has_field_coupling)
        .def_readwrite("H1_pre_field", &Evaluator::H1_pre_field)
        .def_readwrite("field_feats_weight", &Evaluator::field_feats_weight)
        .def_readwrite("field_linear_weight", &Evaluator::field_linear_weight)
        .def_readwrite("electric_field_adj", &Evaluator::electric_field_adj)
        .def_readwrite("electric_field_hessian", &Evaluator::electric_field_hessian)
        .def_readwrite("electric_field_force_derivative", &Evaluator::electric_field_force_derivative)
        .def_readwrite("Phi1", &Evaluator::Phi1)
        .def_readwrite("Phi1_adj", &Evaluator::dPhi1)
        .def_readwrite("A1", &Evaluator::A1)
        .def_readwrite("A1_adj", &Evaluator::A1_adj)
        .def_readwrite("M1", &Evaluator::M1)
        .def_readwrite("M1_adj", &Evaluator::M1_adj)
        .def_readwrite("H2", &Evaluator::H2)
        .def_readwrite("H2_adj", &Evaluator::H2_adj)
        .def("compute_node_energies_forces",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_node_energies_forces(
                           num_nodes,
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()));
            })
        .def("compute_node_energies_forces_field",
            [](Evaluator& self, const int num_nodes,
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
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_electric_field_hessian",
            [](Evaluator& self, const int num_nodes,
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
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_electric_field_force_derivative",
            [](Evaluator& self, const int num_nodes,
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
                           std::span<const int>(node_types.data(), node_types.size()),
                           std::span<const int>(num_neigh.data(), num_neigh.size()),
                           std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                           std::span<const int>(neigh_types.data(), neigh_types.size()),
                           std::span<const double>(xyz.data(), xyz.size()),
                           std::span<const double>(r.data(), r.size()),
                           std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_R0",
            [] (Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_R1",
            [] (Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_R1(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_Y",
            [](Evaluator& self, ContiguousDoubleArray xyz) {
                self.compute_Y(std::span<const double>(xyz.data(), xyz.size()));
            })
        .def("compute_A0",
            [](Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types) {
                self.compute_A0(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()));
            })
        .def("reverse_A0",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_types,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r) {
                self.reverse_A0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()),
                                std::span<const int>(num_neigh.data(), num_neigh.size()),
                                std::span<const int>(neigh_types.data(), neigh_types.size()),
                                std::span<const double>(xyz.data(), xyz.size()),
                                std::span<const double>(r.data(), r.size()));
            })
        .def("compute_A0_scaled",
            [](Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A0_scaled",
            [](Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A0_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M0",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_M0",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.reverse_M0(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H1", &Evaluator::compute_H1)
        .def("reverse_H1", &Evaluator::reverse_H1)
        .def("compute_field_H1",
            [](Evaluator& self, const int num_nodes, ContiguousDoubleArray electric_field) {
                self.compute_field_H1(
                    num_nodes,
                    std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("reverse_field_H1",
            [](Evaluator& self, const int num_nodes, ContiguousDoubleArray electric_field) {
                self.reverse_field_H1(
                    num_nodes,
                    std::span<const double>(electric_field.data(), electric_field.size()));
            })
        .def("compute_Phi1",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices) {
                self.compute_Phi1(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()));
            })
        .def("reverse_Phi1",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray num_neigh,
                           ContiguousIntArray neigh_indices,
                           ContiguousDoubleArray xyz,
                           ContiguousDoubleArray r,
                           bool zero_dxyz,
                           bool zero_H1_adj) {
                self.reverse_Phi1(num_nodes,
                                  std::span<const int>(num_neigh.data(), num_neigh.size()),
                                  std::span<const int>(neigh_indices.data(), neigh_indices.size()),
                                  std::span<const double>(xyz.data(), xyz.size()),
                                  std::span<const double>(r.data(), r.size()),
                                  zero_dxyz,
                                  zero_H1_adj);
            })
        .def("compute_A1", &Evaluator::compute_A1)
        .def("reverse_A1", &Evaluator::reverse_A1)
        .def("compute_A1_scaled",
            [](Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.compute_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("reverse_A1_scaled",
            [](Evaluator& self,
                    const int num_nodes,
                    ContiguousIntArray node_types,
                    ContiguousIntArray num_neigh,
                    ContiguousIntArray neigh_types,
                    ContiguousDoubleArray xyz,
                    ContiguousDoubleArray r) {
                prepare_active_types(self, node_types, neigh_types);
                self.reverse_A1_scaled(
                    num_nodes,
                    std::span<const int>(node_types.data(), node_types.size()),
                    std::span<const int>(num_neigh.data(), num_neigh.size()),
                    std::span<const int>(neigh_types.data(), neigh_types.size()),
                    std::span<const double>(xyz.data(), xyz.size()),
                    std::span<const double>(r.data(), r.size()));
            })
        .def("compute_M1",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })

        .def("reverse_M1",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.reverse_M1(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("compute_H2",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()));
            })
        .def("reverse_H2",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types,
                           bool zero_H1_adj) {
                self.reverse_H2(num_nodes,
                                std::span<const int>(node_types.data(), node_types.size()),
                                zero_H1_adj);
            })
        .def("compute_readouts",
            [](Evaluator& self, const int num_nodes,
                           ContiguousIntArray node_types) {
                self.compute_readouts(num_nodes,
                                      std::span<const int>(node_types.data(), node_types.size()));
            });
}

void bind_mace(py::module_ &m)
{
    bind_mace_cpu<double>(m, "MACE");
    bind_mace_cpu<float>(m, "MACEFloat");
}
