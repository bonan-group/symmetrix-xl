#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "les_monopole.hpp"
#include "tools_kokkos.hpp"

namespace py = pybind11;

namespace {
using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;

template <typename Precision>
void bind_les_monopole(py::module_& module, const char* name)
{
    using Operator = symmetrix::les::LESMonopoleKokkos<Precision>;
    py::class_<Operator>(module, name)
        .def(py::init([] (const double sigma, const double dl,
                          const bool remove_self_interaction) {
            if (!_kokkos_is_initialized())
                _init_kokkos();
            return std::make_unique<Operator>(
                sigma, dl, remove_self_interaction);
        }), py::arg("sigma"), py::arg("dl"),
             py::arg("remove_self_interaction"))
        .def("compute_nonperiodic", [] (Operator& self,
                                         const Array& charges,
                                         const Array& positions) {
            if (charges.ndim() != 1 || positions.ndim() != 2 || positions.shape(1) != 3
                || positions.shape(0) != charges.shape(0))
                throw std::invalid_argument("charges and positions must have shapes (N,) and (N, 3).");
            const auto count = static_cast<std::size_t>(charges.shape(0));
            const auto result = self.compute_nonperiodic(
                std::vector<double>(charges.data(), charges.data()+count),
                std::vector<double>(
                    positions.data(), positions.data()+positions.size()));
            py::array_t<double> atomic_energies(
                {static_cast<py::ssize_t>(count)});
            py::array_t<double> force(py::array::ShapeContainer{
                static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(3)});
            py::array_t<double> adjoints({static_cast<py::ssize_t>(count)});
            std::copy(result.atomic_energies.begin(), result.atomic_energies.end(), atomic_energies.mutable_data());
            std::copy(result.forces.begin(), result.forces.end(), force.mutable_data());
            std::copy(result.charge_adjoints.begin(), result.charge_adjoints.end(), adjoints.mutable_data());
            py::dict output;
            output["energy"] = result.energy;
            output["atomic_energies"] = atomic_energies;
            output["forces"] = force;
            output["charge_adjoints"] = adjoints;
            return output;
        })
        .def("compute_periodic", [] (Operator& self, const Array& charges,
                                      const Array& positions, const Array& cell) {
            if (charges.ndim() != 1 || positions.ndim() != 2 || positions.shape(1) != 3
                || positions.shape(0) != charges.shape(0) || cell.ndim() != 2
                || cell.shape(0) != 3 || cell.shape(1) != 3)
                throw std::invalid_argument("charges, positions, and cell must have shapes (N,), (N, 3), and (3, 3).");
            const auto count = static_cast<std::size_t>(charges.shape(0));
            const auto result = self.compute_periodic(
                std::vector<double>(charges.data(), charges.data()+count),
                std::vector<double>(
                    positions.data(), positions.data()+positions.size()),
                std::vector<double>(cell.data(), cell.data()+9));
            py::array_t<double> atomic_energies({static_cast<py::ssize_t>(count)});
            py::array_t<double> force(py::array::ShapeContainer{
                static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(3)});
            py::array_t<double> adjoints({static_cast<py::ssize_t>(count)});
            py::array_t<double> strain(py::array::ShapeContainer{3, 3});
            std::copy(result.atomic_energies.begin(), result.atomic_energies.end(), atomic_energies.mutable_data());
            std::copy(result.forces.begin(), result.forces.end(), force.mutable_data());
            std::copy(result.charge_adjoints.begin(), result.charge_adjoints.end(), adjoints.mutable_data());
            std::copy(result.strain_derivative.begin(), result.strain_derivative.end(), strain.mutable_data());
            py::dict output;
            output["energy"] = result.energy;
            output["atomic_energies"] = atomic_energies;
            output["forces"] = force;
            output["charge_adjoints"] = adjoints;
            output["strain_derivative"] = strain;
            return output;
        })
        .def("_compute_periodic_device_prefix_for_testing",
            [] (Operator& self, const Array& charges, const Array& positions,
                const Array& cell, const std::size_t count) {
                if (charges.ndim() != 1 || positions.ndim() != 2
                    || positions.shape(1) != 3
                    || positions.shape(0) != charges.shape(0)
                    || count > static_cast<std::size_t>(charges.shape(0))
                    || cell.ndim() != 2 || cell.shape(0) != 3
                    || cell.shape(1) != 3) {
                    throw std::invalid_argument(
                        "charges, positions, and cell must have shapes (capacity,), "
                        "(capacity, 3), and (3, 3), with count <= capacity.");
                }
                const auto capacity = static_cast<std::size_t>(charges.shape(0));
                typename Operator::execution_space execution_space;
                self.prepare_periodic(
                    execution_space,
                    std::vector<double>(cell.data(), cell.data() + 9));
                Kokkos::View<double*> charge_view(
                    "LES retained-capacity test charges", capacity);
                Kokkos::View<double*> position_view(
                    "LES retained-capacity test positions", 3 * capacity);
                auto charge_host = Kokkos::create_mirror_view(charge_view);
                auto position_host = Kokkos::create_mirror_view(position_view);
                std::copy(
                    charges.data(), charges.data() + capacity, charge_host.data());
                std::copy(
                    positions.data(), positions.data() + 3 * capacity,
                    position_host.data());
                Kokkos::deep_copy(execution_space, charge_view, charge_host);
                Kokkos::deep_copy(execution_space, position_view, position_host);
                const auto result = self.compute_periodic_device(
                    execution_space, charge_view, position_view, count);
                const auto energy = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), result.energy);
                const auto atomic = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), Kokkos::subview(
                        result.atomic_energies,
                        std::make_pair(std::size_t(0), count)));
                const auto force = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), Kokkos::subview(
                        result.forces,
                        std::make_pair(std::size_t(0), 3 * count)));
                const auto adjoints = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), Kokkos::subview(
                        result.charge_adjoints,
                        std::make_pair(std::size_t(0), count)));
                const auto strain = Kokkos::create_mirror_view_and_copy(
                    Kokkos::HostSpace(), result.strain_derivative);
                py::array_t<double> atomic_energies(
                    {static_cast<py::ssize_t>(count)});
                py::array_t<double> forces(py::array::ShapeContainer{
                    static_cast<py::ssize_t>(count), static_cast<py::ssize_t>(3)});
                py::array_t<double> charge_adjoints(
                    {static_cast<py::ssize_t>(count)});
                py::array_t<double> strain_derivative(
                    py::array::ShapeContainer{3, 3});
                std::copy(atomic.data(), atomic.data() + count,
                          atomic_energies.mutable_data());
                std::copy(force.data(), force.data() + 3 * count,
                          forces.mutable_data());
                std::copy(adjoints.data(), adjoints.data() + count,
                          charge_adjoints.mutable_data());
                std::copy(strain.data(), strain.data() + 9,
                          strain_derivative.mutable_data());
                py::dict output;
                output["energy"] = energy();
                output["atomic_energies"] = atomic_energies;
                output["forces"] = forces;
                output["charge_adjoints"] = charge_adjoints;
                output["strain_derivative"] = strain_derivative;
                return output;
            }, py::arg("charges"), py::arg("positions"), py::arg("cell"),
            py::arg("count"))
        .def("configure_particle_mesh_for_testing",
            &Operator::configure_particle_mesh_for_testing,
            py::arg("nx"), py::arg("ny"), py::arg("nz"),
            py::arg("spline_order"))
        .def("use_explicit_periodic_for_testing",
            &Operator::use_explicit_periodic_for_testing)
        .def_property_readonly("diagnostics", [] (const Operator& self) {
            const auto diagnostics = self.diagnostics();
            py::dict output;
            output["reciprocal_candidate_count"] =
                diagnostics.reciprocal_candidate_count;
            output["reciprocal_vector_count"] =
                diagnostics.reciprocal_vector_count;
            output["logical_extent_bytes"] = diagnostics.logical_extent_bytes;
            output["allocated_capacity_bytes"] =
                diagnostics.allocated_capacity_bytes;
            output["active_index_capacity_bytes"] =
                diagnostics.active_index_capacity_bytes;
            output["index_pack_count"] = diagnostics.index_pack_count;
            output["table_rebuilds"] = diagnostics.table_rebuilds;
            output["geometry_updates"] = diagnostics.geometry_updates;
            output["active_bytes"] = diagnostics.active_bytes;
            output["capacity_bytes"] = diagnostics.capacity_bytes;
            output["periodic_algorithm"] =
                diagnostics.periodic_algorithm
                    == symmetrix::les::LESPeriodicAlgorithm::particle_mesh
                ? "particle_mesh"
                : "explicit_sum";
            output["particle_count_fences"] =
                diagnostics.particle_mesh_particle_count_fences;
            return output;
        })
        .def("_active_reciprocal_indices_for_testing",
            &Operator::active_reciprocal_indices_for_testing);
}
}

void bind_les_monopole(py::module_& module)
{
    bind_les_monopole<double>(module, "_LESMonopoleKokkos");
    bind_les_monopole<float>(module, "_LESMonopoleKokkosFloat");
}
