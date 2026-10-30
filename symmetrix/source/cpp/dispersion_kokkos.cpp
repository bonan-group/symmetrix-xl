#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "d2_kokkos.hpp"
#include "d3_kokkos.hpp"
#include "dispersion_graph_kokkos.hpp"

#include <Kokkos_Core.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using Graph = symmetrix::dispersion::DispersionGraphKokkos;
using IntegerArray = py::array_t<long long,py::array::c_style|py::array::forcecast>;
using DoubleArray = py::array_t<double,py::array::c_style|py::array::forcecast>;

std::vector<std::string> checked_result_properties(
    const py::object& properties,
    const std::initializer_list<const char*> supported)
{
    std::vector<std::string> requested;
    if (properties.is_none()) {
        requested.reserve(supported.size());
        for (const char* name : supported)
            requested.emplace_back(name);
        return requested;
    }
    requested = properties.cast<std::vector<std::string>>();
    for (const auto& name : requested) {
        const auto found = std::find_if(
            supported.begin(),supported.end(),[&name] (const char* candidate) {
                return name == candidate;
            });
        if (found == supported.end())
            throw std::invalid_argument(
                "Unsupported dispersion result property '"+name+"'.");
    }
    return requested;
}

bool requests(
    const std::vector<std::string>& properties, const std::string& name)
{
    return std::find(properties.begin(),properties.end(),name) != properties.end();
}

std::vector<int> checked_integers(
    const IntegerArray& values, const std::string& description)
{
    std::vector<int> result;
    result.reserve(values.size());
    for (py::ssize_t index=0; index<values.size(); ++index) {
        const long long value = values.data()[index];
        if (value < std::numeric_limits<int>::min()
                || value > std::numeric_limits<int>::max())
            throw std::overflow_error(
                "Dispersion graph "+description+" exceeds 32-bit indexing.");
        result.push_back(static_cast<int>(value));
    }
    return result;
}

template<class View>
py::array_t<typename View::non_const_value_type> export_1d(
    const View& view, const std::size_t count)
{
    using Value = typename View::non_const_value_type;
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),view);
    py::array_t<Value> result(count);
    std::copy(host.data(),host.data()+count,result.mutable_data());
    return result;
}

template<std::size_t Columns, class View>
py::array_t<typename View::non_const_value_type> export_packed_fixed_2d(
    const View& view, const std::size_t count, const char* label)
{
    using Value = typename View::non_const_value_type;
    using PackedView = Kokkos::View<
        Value*[Columns],Kokkos::LayoutRight,typename View::memory_space>;
    PackedView packed(std::string(label),count);
    Kokkos::parallel_for(
        label,
        Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
            0,count*Columns),
        KOKKOS_LAMBDA (const std::size_t index) {
            const std::size_t row = index/Columns;
            const std::size_t column = index%Columns;
            packed(row,column) = view(row,column);
        });
    auto host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace(),packed);
    py::array_t<Value> result(
        std::vector<py::ssize_t>{
            static_cast<py::ssize_t>(count),
            static_cast<py::ssize_t>(Columns)});
    for (std::size_t row=0; row<count; ++row)
        for (std::size_t column=0; column<Columns; ++column)
            result.mutable_data()[Columns*row+column] =
                host(row,column);
    return result;
}

template<class View>
py::array_t<typename View::non_const_value_type> export_2d3(
    const View& view, const std::size_t count)
{
    using Value = typename View::non_const_value_type;
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),view);
    py::array_t<Value> result(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(count),3});
    for (std::size_t row=0; row<count; ++row)
        for (std::size_t column=0; column<3; ++column)
            result.mutable_data()[std::size_t(3)*row+column] =
                host(row,column);
    return result;
}

template<class View>
py::array_t<typename View::non_const_value_type> export_2d6(
    const View& view, const std::size_t count)
{
    using Value = typename View::non_const_value_type;
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),view);
    py::array_t<Value> result(
        std::vector<py::ssize_t>{static_cast<py::ssize_t>(count),6});
    for (std::size_t row=0; row<count; ++row)
        for (std::size_t column=0; column<6; ++column)
            result.mutable_data()[std::size_t(6)*row+column] =
                host(row,column);
    return result;
}

py::dict export_graph(const Graph& graph)
{
    if (!graph.prepared())
        throw std::logic_error(
            "Dispersion graph must be prepared before export.");
    py::dict result;
    const auto nodes = static_cast<std::size_t>(graph.num_nodes());
    const auto edges = graph.num_edges();
    result["atomic_numbers"] = export_1d(graph.atomic_numbers(),nodes);
    result["receiver_offsets"] = export_1d(graph.receiver_offsets(),nodes+1);
    result["receivers"] = export_1d(graph.receivers(),edges);
    result["sources"] = export_1d(graph.sources(),edges);
    result["shifts"] = export_2d3(graph.shifts(),edges);
    result["positions"] = export_2d3(graph.positions(),nodes);
    result["cell"] = export_2d3(graph.cell(),3);
    result["edge_vectors"] = export_2d3(graph.edge_vectors(),edges);
    result["distances"] = export_1d(graph.distances(),edges);
    return result;
}

py::dict graph_diagnostics(const Graph& graph)
{
    const auto diagnostics = graph.diagnostics();
    py::dict result;
    result["graph_backend"] = diagnostics.graph_backend;
    result["generation"] = diagnostics.generation;
    result["geometry_update_count"] = diagnostics.geometry_update_count;
    result["num_nodes"] = diagnostics.num_nodes;
    result["num_edges"] = diagnostics.num_edges;
    result["node_capacity"] = diagnostics.node_capacity;
    result["offset_capacity"] = diagnostics.offset_capacity;
    result["edge_capacity"] = diagnostics.edge_capacity;
    result["build_cutoff"] = diagnostics.build_cutoff;
    result["view_allocation_count"] = diagnostics.view_allocation_count;
    return result;
}

py::dict active_extents(const Graph& graph)
{
    if (!graph.prepared())
        throw std::logic_error(
            "Dispersion graph must be prepared before accessing views.");
    py::dict result;
    result["atomic_numbers"] = graph.atomic_numbers().extent(0);
    result["receiver_offsets"] = graph.receiver_offsets().extent(0);
    result["receivers"] = graph.receivers().extent(0);
    result["sources"] = graph.sources().extent(0);
    result["shifts"] = graph.shifts().extent(0);
    result["positions"] = graph.positions().extent(0);
    result["cell"] = graph.cell().extent(0);
    result["edge_vectors"] = graph.edge_vectors().extent(0);
    result["distances"] = graph.distances().extent(0);
    return result;
}

template<class Precision>
void bind_d2_kokkos(py::module_& module, const char* name)
{
    using Evaluator = symmetrix::dispersion::D2Kokkos<Precision>;
    auto diagnostics = [] (const Evaluator& self) {
        const auto values = self.graph_diagnostics();
        py::dict result;
        result["graph_backend"] = values.graph_backend;
        result["generation"] = values.generation;
        result["geometry_update_count"] = values.geometry_update_count;
        result["num_nodes"] = values.num_nodes;
        result["num_edges"] = values.num_edges;
        result["node_capacity"] = values.node_capacity;
        result["offset_capacity"] = values.offset_capacity;
        result["edge_capacity"] = values.edge_capacity;
        result["build_cutoff"] = values.build_cutoff;
        result["view_allocation_count"] = values.view_allocation_count;
        return result;
    };
    py::class_<Evaluator>(module,name)
        .def(py::init<std::string,double>(),
            py::arg("functional"),py::arg("pair_cutoff"))
        .def(
            "prepare_periodic_graph",
            [] (Evaluator& self, const IntegerArray& atomic_numbers,
                const DoubleArray& positions, const DoubleArray& cell,
                const DoubleArray& inverse_cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                self.prepare_periodic_graph(
                    numbers,
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("positions"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("build_cutoff"))
        .def(
            "prepare_host_graph",
            [] (Evaluator& self, const IntegerArray& atomic_numbers,
                const IntegerArray& receiver_offsets,
                const IntegerArray& sources, const IntegerArray& shifts,
                const DoubleArray& reference_positions,
                const DoubleArray& cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                const auto offsets = checked_integers(
                    receiver_offsets,"32-bit edge indexing");
                const auto source_indices = checked_integers(
                    sources,"source index");
                const auto image_shifts = checked_integers(shifts,"shift");
                self.prepare_host_graph(
                    numbers,offsets,source_indices,image_shifts,
                    std::span<const double>(
                        reference_positions.data(),reference_positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("receiver_offsets"),
            py::arg("sources"),py::arg("shifts"),
            py::arg("reference_positions"),py::arg("cell"),
            py::arg("build_cutoff"))
        .def(
            "update_geometry",
            [] (Evaluator& self, const DoubleArray& positions,
                const DoubleArray& cell, const DoubleArray& inverse_cell) {
                self.update_geometry(
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()));
            },
            py::arg("positions"),py::arg("cell"),py::arg("inverse_cell"))
        .def("compute",&Evaluator::compute,
            py::arg("compute_forces"),py::arg("compute_stress"))
        .def(
            "result",
            [] (const Evaluator& self, const py::object& properties) {
                const auto requested = checked_result_properties(
                    properties,
                    {"energy","atomic_energies","forces","stress",
                     "graph_generation"});
                const auto& native = self.result();
                auto available_requested = requested;
                if (properties.is_none()) {
                    available_requested.erase(
                        std::remove_if(
                            available_requested.begin(),available_requested.end(),
                            [&self] (const std::string& name) {
                                return (name == "forces" && !self.forces_available())
                                    || (name == "stress" && !self.stress_available());
                            }),
                        available_requested.end());
                } else {
                    if (requests(requested,"forces") && !self.forces_available())
                        throw std::logic_error(
                            "D2 forces are unavailable for this compute generation.");
                    if (requests(requested,"stress") && !self.stress_available())
                        throw std::logic_error(
                            "D2 stress is unavailable for this compute generation.");
                }
                const auto graph = self.graph_diagnostics();
                py::dict result;
                if (requests(available_requested,"energy"))
                    result["energy"] = native.energy;
                if (requests(available_requested,"atomic_energies"))
                    result["atomic_energies"] = export_1d(
                        native.atomic_energies,
                        static_cast<std::size_t>(graph.num_nodes));
                if (requests(available_requested,"forces"))
                    result["forces"] = export_2d3(
                        native.forces,static_cast<std::size_t>(graph.num_nodes));
                if (requests(available_requested,"stress"))
                    result["stress"] = export_1d(native.stress,std::size_t(6));
                if (requests(available_requested,"graph_generation"))
                    result["graph_generation"] = graph.generation;
                return result;
            },
            py::arg("properties")=py::none())
        .def("graph_diagnostics",diagnostics)
        .def("diagnostics",diagnostics);
}

template<class Precision>
void bind_d3_kokkos(py::module_& module, const char* name)
{
    using Evaluator = symmetrix::dispersion::D3Kokkos<Precision>;
    auto diagnostics = [] (const Evaluator& self) {
        const auto values = self.graph_diagnostics();
        py::dict result;
        result["graph_backend"] = values.graph_backend;
        result["generation"] = values.generation;
        result["geometry_update_count"] = values.geometry_update_count;
        result["num_nodes"] = values.num_nodes;
        result["num_edges"] = values.num_edges;
        result["node_capacity"] = values.node_capacity;
        result["offset_capacity"] = values.offset_capacity;
        result["edge_capacity"] = values.edge_capacity;
        result["build_cutoff"] = values.build_cutoff;
        result["view_allocation_count"] = self.view_allocation_count();
        result["reference_weight_active_atoms"] =
            self.reference_weight_active_atoms();
        result["reference_weight_capacity_atoms"] =
            self.reference_weight_capacity_atoms();
        result["reference_weight_workspace_bytes"] =
            self.reference_weight_workspace_bytes();
        result["coordination_cache_active_edges"] =
            self.coordination_cache_active_edges();
        result["coordination_cache_capacity_edges"] =
            self.coordination_cache_capacity_edges();
        result["coordination_cache_workspace_bytes"] =
            self.coordination_cache_workspace_bytes();
        result["d3_workspace_bytes"] = self.d3_workspace_bytes();
        return result;
    };
    py::class_<Evaluator>(module,name)
        .def(py::init<std::string,std::string,double,double>(),
            py::arg("damping"),py::arg("functional"),
            py::arg("pair_cutoff"),py::arg("coordination_cutoff"))
        .def(
            "prepare_periodic_graph",
            [] (Evaluator& self, const IntegerArray& atomic_numbers,
                const DoubleArray& positions, const DoubleArray& cell,
                const DoubleArray& inverse_cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                self.prepare_periodic_graph(
                    numbers,
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("positions"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("build_cutoff"))
        .def(
            "prepare_host_graph",
            [] (Evaluator& self, const IntegerArray& atomic_numbers,
                const IntegerArray& receiver_offsets,
                const IntegerArray& sources, const IntegerArray& shifts,
                const DoubleArray& reference_positions,
                const DoubleArray& cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                const auto offsets = checked_integers(
                    receiver_offsets,"32-bit edge indexing");
                const auto source_indices = checked_integers(
                    sources,"source index");
                const auto image_shifts = checked_integers(shifts,"shift");
                self.prepare_host_graph(
                    numbers,offsets,source_indices,image_shifts,
                    std::span<const double>(
                        reference_positions.data(),reference_positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("receiver_offsets"),
            py::arg("sources"),py::arg("shifts"),
            py::arg("reference_positions"),py::arg("cell"),
            py::arg("build_cutoff"))
        .def(
            "update_geometry",
            [] (Evaluator& self, const DoubleArray& positions,
                const DoubleArray& cell, const DoubleArray& inverse_cell) {
                self.update_geometry(
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()));
            },
            py::arg("positions"),py::arg("cell"),py::arg("inverse_cell"))
        .def("compute",&Evaluator::compute,
            py::arg("compute_forces"),py::arg("compute_stress"))
        .def(
            "_interpolation_diagnostics",
            [] (const Evaluator& self, const int receiver_atomic_number,
                const int source_atomic_number,
                const double receiver_coordination,
                const double source_coordination) {
                const auto values = self.interpolation_diagnostics(
                    receiver_atomic_number,source_atomic_number,
                    receiver_coordination,source_coordination);
                py::dict result;
                result["c6"] = values[0];
                result["receiver_derivative"] = values[1];
                result["source_derivative"] = values[2];
                result["factorized_c6"] = values[3];
                result["factorized_receiver_derivative"] = values[4];
                result["factorized_source_derivative"] = values[5];
                return result;
            },
            py::arg("receiver_atomic_number"),
            py::arg("source_atomic_number"),
            py::arg("receiver_coordination"),
            py::arg("source_coordination"))
        .def(
            "_pair_energy_diagnostics",
            &Evaluator::pair_energy_diagnostics,
            py::arg("receiver_atomic_number"),
            py::arg("source_atomic_number"),py::arg("distance"),
            py::arg("receiver_coordination"),
            py::arg("source_coordination"))
        .def(
            "_coordination_switch_cache_diagnostics",
            [] (const Evaluator& self) {
                const auto values = self.coordination_switch_values();
                const auto derivatives =
                    self.coordination_radial_derivatives();
                const auto active = values.extent(0);
                const auto capacity = self.coordination_switch_capacity();
                py::dict result;
                result["values"] = export_1d(values,active);
                result["radial_derivatives"] =
                    export_1d(derivatives,active);
                result["capacity"] = capacity;
                result["workspace_bytes"] =
                    2*capacity*sizeof(Precision);
                return result;
            })
        .def(
            "_coordination_accumulation_diagnostics",
            [] (Evaluator& self, const std::string& selector) {
                const bool use_team = selector == "team";
                if (!use_team && selector != "serial")
                    throw std::invalid_argument(
                        "coordination diagnostic selector must be 'team' or "
                        "'serial'.");
                self.compute_coordination_diagnostics(use_team);
                const auto graph = self.graph_diagnostics();
                const auto nodes = static_cast<std::size_t>(graph.num_nodes);
                const auto edges = static_cast<std::size_t>(graph.num_edges);
                py::dict result;
                result["execution_path"] =
                    use_team ? "team" : "serial_oracle";
                result["coordination_numbers"] = export_1d(
                    self.coordination_numbers_diagnostics(),nodes);
                result["values"] = export_1d(
                    self.coordination_switch_values_diagnostics(),edges);
                result["radial_derivatives"] = export_1d(
                    self.coordination_radial_derivatives_diagnostics(),edges);
                result["team_launch_count"] =
                    self.coordination_team_launch_count();
                result["serial_fallback_launch_count"] =
                    self.coordination_serial_fallback_launch_count();
                return result;
            },
            py::arg("selector"))
        .def(
            "_coordination_path_diagnostics",
            [] (const Evaluator& self) {
                py::dict result;
                result["execution_path"] =
                    self.coordination_production_path();
                result["team_launch_count"] =
                    self.coordination_team_launch_count();
                result["serial_fallback_launch_count"] =
                    self.coordination_serial_fallback_launch_count();
                return result;
            })
        .def(
            "_stress_reduction_diagnostics",
            [] (const Evaluator& self) {
                py::dict result;
                result["stress_reduction_launch_count"] =
                    self.stress_reduction_launch_count();
                return result;
            })
        .def(
            "_pair_accumulation_diagnostics",
            [] (Evaluator& self, const std::string& selector,
                const bool compute_forces, const bool compute_stress) {
                const bool use_team = selector == "team";
                if (!use_team && selector != "serial")
                    throw std::invalid_argument(
                        "pair diagnostic selector must be 'team' or "
                        "'serial'.");
                self.compute_pair_diagnostics(
                    use_team,compute_forces,compute_stress);
                const auto graph = self.graph_diagnostics();
                const auto nodes = static_cast<std::size_t>(graph.num_nodes);
                py::dict result;
                result["execution_path"] =
                    use_team ? "team" : "serial_oracle";
                result["atomic_energies"] = export_1d(
                    self.atomic_energies_diagnostics(),nodes);
                result["energy_coordination_derivatives"] = export_1d(
                    self.energy_coordination_derivatives_diagnostics(),nodes);
                result["forces"] = export_2d3(
                    self.forces_diagnostics(),nodes);
                result["virial"] = export_2d6(
                    self.virial_diagnostics(),nodes);
                return result;
            },
            py::arg("selector"),py::arg("compute_forces"),
            py::arg("compute_stress"))
        .def(
            "_adjoint_accumulation_diagnostics",
            [] (Evaluator& self, const std::string& selector) {
                const bool use_team = selector == "team";
                if (!use_team && selector != "serial")
                    throw std::invalid_argument(
                        "adjoint diagnostic selector must be 'team' or "
                        "'serial'.");
                self.compute_adjoint_diagnostics(use_team);
                const auto graph = self.graph_diagnostics();
                const auto nodes = static_cast<std::size_t>(graph.num_nodes);
                py::dict result;
                result["execution_path"] =
                    use_team ? "team" : "serial_oracle";
                result["forces"] = export_packed_fixed_2d<3>(
                    self.forces_diagnostics(),nodes,
                    "D3 adjoint force diagnostic export");
                result["virial"] = export_packed_fixed_2d<6>(
                    self.virial_diagnostics(),nodes,
                    "D3 adjoint virial diagnostic export");
                return result;
            },
            py::arg("selector"))
        .def(
            "result",
            [] (const Evaluator& self, const py::object& properties) {
                const auto requested = checked_result_properties(
                    properties,
                    {"energy","atomic_energies","forces","stress",
                     "coordination_numbers",
                     "energy_coordination_derivatives","graph_generation"});
                const auto& native = self.result();
                auto available_requested = requested;
                if (properties.is_none()) {
                    available_requested.erase(
                        std::remove_if(
                            available_requested.begin(),available_requested.end(),
                            [&self] (const std::string& name) {
                                return (name == "forces" && !self.forces_available())
                                    || (name == "stress" && !self.stress_available());
                            }),
                        available_requested.end());
                } else {
                    if (requests(requested,"forces") && !self.forces_available())
                        throw std::logic_error(
                            "D3 forces are unavailable for this compute generation.");
                    if (requests(requested,"stress") && !self.stress_available())
                        throw std::logic_error(
                            "D3 stress is unavailable for this compute generation.");
                }
                const auto graph = self.graph_diagnostics();
                const auto nodes = static_cast<std::size_t>(graph.num_nodes);
                py::dict result;
                if (requests(available_requested,"energy"))
                    result["energy"] = native.energy;
                if (requests(available_requested,"atomic_energies"))
                    result["atomic_energies"] = export_1d(
                        native.atomic_energies,nodes);
                if (requests(available_requested,"forces"))
                    result["forces"] = export_2d3(native.forces,nodes);
                if (requests(available_requested,"stress"))
                    result["stress"] = export_1d(native.stress,std::size_t(6));
                if (requests(available_requested,"coordination_numbers"))
                    result["coordination_numbers"] = export_1d(
                        self.coordination_numbers(),nodes);
                if (requests(available_requested,"energy_coordination_derivatives"))
                    result["energy_coordination_derivatives"] = export_1d(
                        self.energy_coordination_derivatives(),nodes);
                if (requests(available_requested,"graph_generation"))
                    result["graph_generation"] = graph.generation;
                return result;
            },
            py::arg("properties")=py::none())
        .def("graph_diagnostics",diagnostics)
        .def("diagnostics",diagnostics);
}

} // namespace

void bind_dispersion_kokkos(py::module_& m)
{
    py::class_<Graph>(m,"_DispersionGraphKokkos")
        .def(py::init<>())
        .def(
            "prepare_periodic",
            [] (Graph& self, const IntegerArray& atomic_numbers,
                const DoubleArray& positions, const DoubleArray& cell,
                const DoubleArray& inverse_cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                self.prepare_periodic(
                    numbers,
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("positions"),py::arg("cell"),
            py::arg("inverse_cell"),py::arg("build_cutoff"))
        .def(
            "prepare_host",
            [] (Graph& self, const IntegerArray& atomic_numbers,
                const IntegerArray& receiver_offsets,
                const IntegerArray& sources, const IntegerArray& shifts,
                const DoubleArray& reference_positions,
                const DoubleArray& cell, const double build_cutoff) {
                const auto numbers = checked_integers(
                    atomic_numbers,"atomic number");
                const auto offsets = checked_integers(
                    receiver_offsets,"32-bit edge indexing");
                const auto source_indices = checked_integers(
                    sources,"source index");
                const auto image_shifts = checked_integers(shifts,"shift");
                self.prepare_host(
                    numbers,offsets,source_indices,image_shifts,
                    std::span<const double>(
                        reference_positions.data(),reference_positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    build_cutoff);
            },
            py::arg("atomic_numbers"),py::arg("receiver_offsets"),
            py::arg("sources"),py::arg("shifts"),
            py::arg("reference_positions"),py::arg("cell"),
            py::arg("build_cutoff"))
        .def(
            "update_geometry",
            [] (Graph& self, const DoubleArray& positions,
                const DoubleArray& cell, const DoubleArray& inverse_cell) {
                self.update_geometry(
                    std::span<const double>(positions.data(),positions.size()),
                    std::span<const double>(cell.data(),cell.size()),
                    std::span<const double>(
                        inverse_cell.data(),inverse_cell.size()));
            },
            py::arg("positions"),py::arg("cell"),py::arg("inverse_cell"))
        .def("export",&export_graph)
        .def("active_extents",&active_extents)
        .def("diagnostics",&graph_diagnostics);
    bind_d2_kokkos<double>(m,"D2Kokkos");
    bind_d2_kokkos<float>(m,"D2KokkosFloat");
    bind_d3_kokkos<double>(m,"D3Kokkos");
    bind_d3_kokkos<float>(m,"D3KokkosFloat");
}
