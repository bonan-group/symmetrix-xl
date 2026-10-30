#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "oam_schema.hpp"

namespace py = pybind11;

void bind_nequip_schema(py::module_& module)
{
    py::class_<OAMRadialSplineContract>(module, "OAMRadialSplineContract")
        .def_readonly("grid_min", &OAMRadialSplineContract::grid_min)
        .def_readonly("cutoff", &OAMRadialSplineContract::cutoff)
        .def_readonly("num_points", &OAMRadialSplineContract::num_points)
        .def_readonly("spacing", &OAMRadialSplineContract::spacing)
        .def_readonly("function_count", &OAMRadialSplineContract::function_count)
        .def_readonly("edge_type_count", &OAMRadialSplineContract::edge_type_count)
        .def_readonly("model_fingerprint", &OAMRadialSplineContract::model_fingerprint)
        .def_readonly("table_fingerprint", &OAMRadialSplineContract::table_fingerprint)
        .def_readonly("precision", &OAMRadialSplineContract::precision)
        .def_readonly("exporter_version", &OAMRadialSplineContract::exporter_version)
        .def_readonly("certification", &OAMRadialSplineContract::certification)
        .def_readonly("certified", &OAMRadialSplineContract::certified);
    py::class_<NequIPOAMLayer>(module, "NequIPOAMLayer")
        .def_readonly("node_irreps_in", &NequIPOAMLayer::node_irreps_in)
        .def_readonly("edge_irreps", &NequIPOAMLayer::edge_irreps)
        .def_readonly("tp_output_irreps", &NequIPOAMLayer::tp_output_irreps)
        .def_readonly("node_irreps_pre_gate", &NequIPOAMLayer::node_irreps_pre_gate)
        .def_readonly("node_irreps_out", &NequIPOAMLayer::node_irreps_out)
        .def_readonly("pre_linear", &NequIPOAMLayer::pre_linear)
        .def_readonly("radial_basis", &NequIPOAMLayer::radial_basis)
        .def_readonly("edge_tensor_product", &NequIPOAMLayer::edge_tensor_product)
        .def_readonly("post_linear", &NequIPOAMLayer::post_linear)
        .def_readonly("self_connection", &NequIPOAMLayer::self_connection)
        .def_readonly("normalization", &NequIPOAMLayer::normalization)
        .def_readonly("gate", &NequIPOAMLayer::gate);
    py::class_<NequIPOAMModelSchema>(module, "NequIPOAMModelSchema")
        .def_readonly("atomic_numbers", &NequIPOAMModelSchema::atomic_numbers)
        .def_readonly("model_atomic_numbers", &NequIPOAMModelSchema::model_atomic_numbers)
        .def_readonly("atomic_number_to_model_type", &NequIPOAMModelSchema::atomic_number_to_model_type)
        .def_readonly("r_cut", &NequIPOAMModelSchema::r_cut)
        .def_readonly("edge_embedding", &NequIPOAMModelSchema::edge_embedding)
        .def_readonly("node_attributes", &NequIPOAMModelSchema::node_attributes)
        .def_readonly("layers", &NequIPOAMModelSchema::layers)
        .def_readonly("readout", &NequIPOAMModelSchema::readout)
        .def_readonly("per_type_shift", &NequIPOAMModelSchema::per_type_shift)
        .def_readonly("per_type_scale", &NequIPOAMModelSchema::per_type_scale)
        .def_readonly("zbl", &NequIPOAMModelSchema::zbl)
        .def_readonly("kokkos_radial", &NequIPOAMModelSchema::kokkos_radial)
        .def_readonly("exporter_fingerprint", &NequIPOAMModelSchema::exporter_fingerprint)
        .def("contract_fingerprint", &NequIPOAMModelSchema::contract_fingerprint)
        .def("generated_architecture_fingerprint",
            &NequIPOAMModelSchema::generated_architecture_fingerprint);
    module.def("load_nequip_oam_schema", &load_nequip_oam_schema);
}
