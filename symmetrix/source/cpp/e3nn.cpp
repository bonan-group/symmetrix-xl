#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "affine_mlp.hpp"
#include "e3nn.hpp"
#include "e3nn_product.hpp"

#ifdef SYMMETRIX_KOKKOS
#include "affine_mlp_kokkos.hpp"
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "tools_kokkos.hpp"
#endif

namespace py = pybind11;

#ifdef SYMMETRIX_KOKKOS
namespace {
void bind_float_e3_primitives(py::module_& module)
{
    using Linear=E3LinearFloatKokkos;
    py::class_<Linear>(module,"E3LinearKokkosFloat")
        .def(py::init([](const std::string& definition) {
            return Linear(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension",&Linear::input_dimension)
        .def_property_readonly("output_dimension",&Linear::output_dimension)
        .def_property_readonly("backend",&Linear::backend)
        .def_property_readonly("workspace_bytes",&Linear::workspace_bytes)
        .def("set_backend",&Linear::set_backend)
        .def("selected_backend",&Linear::selected_backend)
        .def("evaluate_batch",[](
            const Linear& self,const std::vector<float>& values,int samples) {
            if(samples<0||values.size()!=static_cast<std::size_t>(samples)
                    *self.input_dimension())
                throw std::invalid_argument(
                    "Float Kokkos e3 linear input dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight> input,output;
            set_kokkos_view(input,values,samples,self.input_dimension());
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate(input,output);
            return view2vector(output);
        })
        .def("evaluate_ir_mul_batch",[](
            const Linear& self,const std::vector<float>& values,int samples) {
            if(samples<0||values.size()!=static_cast<std::size_t>(samples)
                    *self.input_dimension())
                throw std::invalid_argument(
                    "Float Kokkos ir-mul e3 linear input dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight> input,output;
            set_kokkos_view(input,values,samples,self.input_dimension());
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate_ir_mul(input,output);
            return view2vector(output);
        })
        .def("reverse_batch",[](
            const Linear& self,const std::vector<float>& seed,int samples) {
            if(samples<0||seed.size()!=static_cast<std::size_t>(samples)
                    *self.output_dimension())
                throw std::invalid_argument(
                    "Float Kokkos e3 linear adjoint dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight> output_adjoint,input_adjoint;
            set_kokkos_view(
                output_adjoint,seed,samples,self.output_dimension());
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            self.reverse(output_adjoint,input_adjoint);
            return view2vector(input_adjoint);
        })
        .def("reverse_ir_mul_batch",[](
            const Linear& self,const std::vector<float>& seed,int samples) {
            if(samples<0||seed.size()!=static_cast<std::size_t>(samples)
                    *self.output_dimension())
                throw std::invalid_argument(
                    "Float Kokkos ir-mul e3 linear adjoint dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight>
                output_adjoint,input_adjoint;
            set_kokkos_view(
                output_adjoint,seed,samples,self.output_dimension());
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            self.reverse_ir_mul(output_adjoint,input_adjoint);
            return view2vector(input_adjoint);
        });

    using Tensor=E3TensorProductFloatKokkos;
    py::class_<Tensor>(module,"E3TensorProductKokkosFloat")
        .def(py::init([](const std::string& definition) {
            return Tensor(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_1_dimension",&Tensor::input_1_dimension)
        .def_property_readonly("input_2_dimension",&Tensor::input_2_dimension)
        .def_property_readonly("output_dimension",&Tensor::output_dimension)
        .def_property_readonly("weight_size",&Tensor::weight_size)
        .def_property_readonly(
            "has_internal_weights",&Tensor::has_internal_weights)
        .def_property_readonly(
            "uses_mh1_fast_path",&Tensor::uses_mh1_fast_path)
        .def_property_readonly("backend",&Tensor::backend)
        .def_property_readonly("execution_backend",&Tensor::execution_backend)
        .def_property_readonly("channel_team_size",&Tensor::channel_team_size)
        .def_property_readonly("harmonic_team_size",&Tensor::harmonic_team_size)
        .def_property_readonly(
            "supports_direct_node_reverse",&Tensor::supports_direct_node_reverse)
        .def("evaluate_batch",[](
            const Tensor& self,const std::vector<float>& input_1,
            const std::vector<float>& input_2,const std::vector<float>& weights,
            int samples) {
            if(samples<0
                ||input_1.size()!=static_cast<std::size_t>(samples)
                    *self.input_1_dimension()
                ||input_2.size()!=static_cast<std::size_t>(samples)
                    *self.input_2_dimension())
                throw std::invalid_argument(
                    "Float Kokkos tensor-product input dimensions are inconsistent.");
            const bool use_internal=weights.empty()&&self.has_internal_weights();
            if(!use_internal&&weights.size()!=static_cast<std::size_t>(samples)
                    *self.weight_size())
                throw std::invalid_argument(
                    "Float Kokkos tensor-product weight dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight> first,second,weight,output;
            set_kokkos_view(first,input_1,samples,self.input_1_dimension());
            set_kokkos_view(second,input_2,samples,self.input_2_dimension());
            set_kokkos_view(
                weight,weights,samples,use_internal?0:self.weight_size());
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate(first,second,weight,output);
            return view2vector(output);
        })
        .def("reverse_batch",[](
            const Tensor& self,const std::vector<float>& input_1,
            const std::vector<float>& input_2,const std::vector<float>& weights,
            const std::vector<float>& seed,int samples) {
            if(samples<0
                ||input_1.size()!=static_cast<std::size_t>(samples)
                    *self.input_1_dimension()
                ||input_2.size()!=static_cast<std::size_t>(samples)
                    *self.input_2_dimension()
                ||seed.size()!=static_cast<std::size_t>(samples)
                    *self.output_dimension())
                throw std::invalid_argument(
                    "Float Kokkos tensor-product reverse dimensions are inconsistent.");
            const bool use_internal=weights.empty()&&self.has_internal_weights();
            if(!use_internal&&weights.size()!=static_cast<std::size_t>(samples)
                    *self.weight_size())
                throw std::invalid_argument(
                    "Float Kokkos tensor-product weight dimensions are inconsistent.");
            Kokkos::View<float**,Kokkos::LayoutRight> first,second,weight;
            Kokkos::View<float**,Kokkos::LayoutRight> output_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> first_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> second_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> weight_adjoint;
            set_kokkos_view(first,input_1,samples,self.input_1_dimension());
            set_kokkos_view(second,input_2,samples,self.input_2_dimension());
            set_kokkos_view(
                weight,weights,samples,use_internal?0:self.weight_size());
            set_kokkos_view(
                output_adjoint,seed,samples,self.output_dimension());
            Kokkos::realloc(first_adjoint,samples,self.input_1_dimension());
            Kokkos::realloc(second_adjoint,samples,self.input_2_dimension());
            Kokkos::realloc(weight_adjoint,samples,self.weight_size());
            self.reverse(
                first,second,weight,output_adjoint,first_adjoint,
                second_adjoint,weight_adjoint);
            return py::make_tuple(
                view2vector(first_adjoint),view2vector(second_adjoint),
                view2vector(weight_adjoint));
        })
        .def("reverse_from_nodes",[](
            const Tensor& self,const std::vector<float>& source_node_values,
            const std::vector<int>& source_indices,int first_edge,
            const std::vector<float>& edge_input_2,
            const std::vector<float>& edge_weights,
            const std::vector<float>& target_node_output_adjoint,
            const std::vector<int>& target_indices,
            const std::vector<float>& initial_source_node_input_adjoint,
            int samples) {
            const int input_dimension=self.input_1_dimension();
            const int output_dimension=self.output_dimension();
            if(samples<0||first_edge<0||input_dimension<=0||output_dimension<=0
                ||source_node_values.size()%input_dimension!=0
                ||target_node_output_adjoint.size()%output_dimension!=0
                ||initial_source_node_input_adjoint.size()
                    !=source_node_values.size()
                ||source_indices.size()
                    <static_cast<std::size_t>(first_edge)+samples
                ||target_indices.size()
                    <static_cast<std::size_t>(first_edge)+samples
                ||edge_input_2.size()!=static_cast<std::size_t>(samples)
                    *self.input_2_dimension()
                ||edge_weights.size()!=static_cast<std::size_t>(samples)
                    *self.weight_size())
                throw std::invalid_argument(
                    "Float Kokkos direct-node tensor-product reverse dimensions "
                    "are inconsistent.");
            const int source_nodes=source_node_values.size()/input_dimension;
            const int target_nodes=
                target_node_output_adjoint.size()/output_dimension;
            const std::size_t edge_end=
                static_cast<std::size_t>(first_edge)+samples;
            for(std::size_t edge=first_edge;edge<edge_end;++edge)
                if(source_indices[edge]<0||source_indices[edge]>=source_nodes
                    ||target_indices[edge]<0||target_indices[edge]>=target_nodes)
                    throw std::invalid_argument(
                        "Float Kokkos direct-node tensor-product reverse indices "
                        "are out of bounds.");
            Kokkos::View<float**,Kokkos::LayoutRight> source_values;
            Kokkos::View<float**,Kokkos::LayoutRight> second,weight,target_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> source_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> second_adjoint;
            Kokkos::View<float**,Kokkos::LayoutRight> weight_adjoint;
            Kokkos::View<int*> sources,targets;
            set_kokkos_view(
                source_values,source_node_values,source_nodes,input_dimension);
            set_kokkos_view(sources,source_indices);
            set_kokkos_view(
                second,edge_input_2,samples,self.input_2_dimension());
            set_kokkos_view(weight,edge_weights,samples,self.weight_size());
            set_kokkos_view(
                target_adjoint,target_node_output_adjoint,
                target_nodes,output_dimension);
            set_kokkos_view(targets,target_indices);
            set_kokkos_view(
                source_adjoint,initial_source_node_input_adjoint,
                source_nodes,input_dimension);
            Kokkos::realloc(
                second_adjoint,samples,self.input_2_dimension());
            Kokkos::realloc(weight_adjoint,samples,self.weight_size());
            if(!self.try_reverse_from_nodes(
                source_values,sources,first_edge,second,weight,target_adjoint,
                targets,source_adjoint,second_adjoint,weight_adjoint))
                throw std::invalid_argument(
                    "Float Kokkos direct-node tensor-product reverse is unsupported.");
            return py::make_tuple(
                view2vector(source_adjoint),view2vector(second_adjoint),
                view2vector(weight_adjoint));
        });
}
}
#endif

void bind_e3nn(py::module_& module)
{
    py::class_<AffineMLP>(module, "AffineMLP")
        .def(py::init([](const std::string& definition) {
            return AffineMLP(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_size", &AffineMLP::input_size)
        .def_property_readonly("output_size", &AffineMLP::output_size)
        .def("supports_conditioned_input", &AffineMLP::supports_conditioned_input)
        .def("first_layer_contribution", &AffineMLP::first_layer_contribution)
        .def("evaluate", &AffineMLP::evaluate)
        .def("evaluate_conditioned", &AffineMLP::evaluate_conditioned)
        .def("evaluate_conditioned_directional", [](
            const AffineMLP& self,
            const std::vector<double>& input,
            const std::vector<double>& input_derivative,
            const std::vector<double>& first_contribution,
            const std::vector<double>& second_contribution) {
            std::vector<double> output,output_derivative;
            self.evaluate_conditioned_with_directional_derivative(
                input,input_derivative,first_contribution,second_contribution,
                output,output_derivative);
            return py::make_tuple(output,output_derivative);
        })
        .def("reverse_conditioned", &AffineMLP::evaluate_gradient_conditioned)
        .def("conditioned_batch", [](
            const AffineMLP& self,
            const std::vector<double>& input,
            int samples,
            int dynamic_input_size,
            const std::vector<double>& row_contributions,
            const std::vector<double>& output_adjoint) {
            AffineMLPBatchTape tape;
            const auto& output = self.evaluate_conditioned_batch(
                input, samples, dynamic_input_size, row_contributions, tape);
            AffineMLPBatchWorkspace workspace;
            std::vector<double> input_adjoint;
            self.reverse_conditioned_batch(
                output_adjoint, tape, input_adjoint, workspace);
            return py::make_tuple(output, input_adjoint);
        })
        .def("reverse", &AffineMLP::evaluate_gradient);

    py::class_<E3Linear>(module, "E3Linear")
        .def(py::init([](const std::string& definition) {
            return E3Linear(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3Linear::input_dimension)
        .def_property_readonly("output_dimension", &E3Linear::output_dimension)
        .def("evaluate", &E3Linear::evaluate)
        .def("reverse", [](const E3Linear& self, const std::vector<double>& output_adjoint) {
            std::vector<double> input_adjoint;
            self.reverse(output_adjoint, input_adjoint);
            return input_adjoint;
        })
        .def("evaluate_batch", [](
            const E3Linear& self,
            const std::vector<double>& input_values,
            int samples) {
            E3LinearBatchWorkspace workspace;
            std::vector<double> output_values;
            self.evaluate_batch(input_values, samples, output_values, workspace);
            return output_values;
        })
        .def("reverse_batch", [](
            const E3Linear& self,
            const std::vector<double>& output_adjoint,
            int samples) {
            E3LinearBatchWorkspace workspace;
            std::vector<double> input_adjoint;
            self.reverse_batch(output_adjoint, samples, input_adjoint, workspace);
            return input_adjoint;
        });

    py::class_<E3TensorProduct>(module, "E3TensorProduct")
        .def(py::init([](const std::string& definition) {
            return E3TensorProduct(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_1_dimension", &E3TensorProduct::input_1_dimension)
        .def_property_readonly("input_2_dimension", &E3TensorProduct::input_2_dimension)
        .def_property_readonly("output_dimension", &E3TensorProduct::output_dimension)
        .def("evaluate", &E3TensorProduct::evaluate,
             py::arg("input_1"), py::arg("input_2"), py::arg("weights") = std::vector<double>{})
        .def("reverse", [](
            const E3TensorProduct& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights,
            const std::vector<double>& output_adjoint) {
            std::vector<double> input_1_adjoint;
            std::vector<double> input_2_adjoint;
            std::vector<double> weights_adjoint;
            self.reverse(input_1, input_2, weights, output_adjoint,
                         input_1_adjoint, input_2_adjoint, weights_adjoint);
            return py::make_tuple(input_1_adjoint, input_2_adjoint, weights_adjoint);
        });

    py::class_<E3ProductBasis>(module, "E3ProductBasis")
        .def(py::init([](const std::string& definition) {
            return E3ProductBasis(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3ProductBasis::input_dimension)
        .def_property_readonly("output_dimension", &E3ProductBasis::output_dimension)
        .def_property_readonly("uses_compiled_plan", &E3ProductBasis::uses_compiled_plan)
        .def_property_readonly("compiled_term_count", &E3ProductBasis::compiled_term_count)
        .def("evaluate", &E3ProductBasis::evaluate)
        .def("reverse", [](
            const E3ProductBasis& self,
            const std::vector<double>& node_features,
            int element,
            const std::vector<double>& output_adjoint) {
            std::vector<double> node_features_adjoint;
            std::vector<double> skip_connection_adjoint;
            self.reverse(node_features, element, output_adjoint,
                         node_features_adjoint, skip_connection_adjoint);
            return py::make_tuple(node_features_adjoint, skip_connection_adjoint);
        })
        .def("evaluate_batch", [](
            const E3ProductBasis& self,
            const std::vector<double>& node_features,
            const std::vector<double>& skip_connections,
            const std::vector<int>& elements,
            int samples) {
            E3ProductBasisBatchWorkspace workspace;
            std::vector<double> output_values;
            self.evaluate_batch(
                node_features, skip_connections, elements, samples,
                output_values, workspace);
            return output_values;
        })
        .def("reverse_batch", [](
            const E3ProductBasis& self,
            const std::vector<double>& node_features,
            const std::vector<int>& elements,
            const std::vector<double>& output_adjoint,
            int samples) {
            E3ProductBasisBatchWorkspace workspace;
            std::vector<double> node_features_adjoint;
            std::vector<double> skip_connection_adjoint;
            self.reverse_batch(
                node_features, elements, output_adjoint, samples,
                node_features_adjoint, skip_connection_adjoint, workspace);
            return py::make_tuple(node_features_adjoint, skip_connection_adjoint);
        });

#ifdef SYMMETRIX_KOKKOS
    bind_float_e3_primitives(module);
    py::class_<AffineMLPKokkos>(module, "AffineMLPKokkos")
        .def(py::init([](const std::string& definition) {
            return AffineMLPKokkos(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_size", &AffineMLPKokkos::input_size)
        .def_property_readonly("output_size", &AffineMLPKokkos::output_size)
        .def("supports_conditioned_input", &AffineMLPKokkos::supports_conditioned_input)
        .def("conditioned_batch", [](
            AffineMLPKokkos& self,
            const std::vector<double>& input,
            int samples,
            int dynamic_input_size,
            const std::vector<double>& row_contributions,
            const std::vector<double>& output_adjoint) {
            if(samples<0||!self.supports_conditioned_input(dynamic_input_size)
                ||input.size()!=static_cast<std::size_t>(samples)*dynamic_input_size
                ||output_adjoint.size()!=static_cast<std::size_t>(samples)*self.output_size())
                throw std::invalid_argument(
                    "Kokkos conditioned affine batch dimensions are inconsistent.");
            const int contribution_width=row_contributions.size()
                /static_cast<std::size_t>(samples?samples:1);
            if(row_contributions.size()
                !=static_cast<std::size_t>(samples)*contribution_width)
                throw std::invalid_argument(
                    "Kokkos conditioned affine contributions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input_view,contribution_view;
            Kokkos::View<double**,Kokkos::LayoutRight> output_view,seed_view,input_adjoint;
            set_kokkos_view(input_view,input,samples,dynamic_input_size);
            set_kokkos_view(
                contribution_view,row_contributions,samples,contribution_width);
            set_kokkos_view(seed_view,output_adjoint,samples,self.output_size());
            Kokkos::realloc(output_view,samples,self.output_size());
            Kokkos::realloc(input_adjoint,samples,dynamic_input_size);
            self.evaluate_conditioned(input_view,contribution_view,output_view);
            self.reverse_from_tape(seed_view,input_adjoint);
            return py::make_tuple(view2vector(output_view),view2vector(input_adjoint));
        });

    py::class_<E3LinearKokkos>(module, "E3LinearKokkos")
        .def(py::init([](const std::string& definition) {
            return E3LinearKokkos(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3LinearKokkos::input_dimension)
        .def_property_readonly("output_dimension", &E3LinearKokkos::output_dimension)
        .def("evaluate_batch", [](
            const E3LinearKokkos& self,
            const std::vector<double>& input_values,
            int samples) {
            if(samples<0||input_values.size()
                !=static_cast<std::size_t>(samples)*self.input_dimension())
                throw std::invalid_argument("Kokkos e3 linear input dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input,output;
            set_kokkos_view(input,input_values,samples,self.input_dimension());
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate(input,output);
            return view2vector(output);
        })
        .def("evaluate_to_packed_batch", [](
            const E3LinearKokkos& self,
            const std::vector<double>& input_values,
            int samples) {
            if(samples<0||input_values.size()
                !=static_cast<std::size_t>(samples)*self.input_dimension())
                throw std::invalid_argument(
                    "Kokkos packed e3 linear input dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input,output;
            set_kokkos_view(input,input_values,samples,self.input_dimension());
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate_to_packed(input,output);
            return view2vector(output);
        })
        .def("reverse_batch", [](
            const E3LinearKokkos& self,
            const std::vector<double>& output_adjoint,
            int samples) {
            if(samples<0||output_adjoint.size()
                !=static_cast<std::size_t>(samples)*self.output_dimension())
                throw std::invalid_argument("Kokkos e3 linear adjoint dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> seed,input_adjoint;
            set_kokkos_view(seed,output_adjoint,samples,self.output_dimension());
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            self.reverse(seed,input_adjoint);
            return view2vector(input_adjoint);
        })
        .def("reverse_packed_to_packed_batch", [](
            const E3LinearKokkos& self,
            const std::vector<double>& packed_output_adjoint,
            int samples) {
            if(samples<0||packed_output_adjoint.size()
                !=static_cast<std::size_t>(samples)*self.output_dimension())
                throw std::invalid_argument(
                    "Kokkos packed e3 linear adjoint dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> seed,input_adjoint;
            set_kokkos_view(
                seed,packed_output_adjoint,samples,self.output_dimension());
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            self.reverse_packed_to_packed(seed,input_adjoint);
            return view2vector(input_adjoint);
        })
        .def("reverse_from_packed_batch", [](
            const E3LinearKokkos& self,
            const std::vector<double>& packed_output_adjoint,
            int samples) {
            if(samples<0||packed_output_adjoint.size()
                !=static_cast<std::size_t>(samples)*self.output_dimension())
                throw std::invalid_argument(
                    "Kokkos packed e3 linear adjoint dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> seed,input_adjoint;
            set_kokkos_view(
                seed,packed_output_adjoint,samples,self.output_dimension());
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            self.reverse_from_packed(seed,input_adjoint);
            return view2vector(input_adjoint);
        });

    py::class_<E3TensorProductKokkos>(module, "E3TensorProductKokkos")
        .def(py::init([](const std::string& definition) {
            return E3TensorProductKokkos(nlohmann::json::parse(definition));
        }))
        .def_property_readonly(
            "input_1_dimension", &E3TensorProductKokkos::input_1_dimension)
        .def_property_readonly(
            "input_2_dimension", &E3TensorProductKokkos::input_2_dimension)
        .def_property_readonly(
            "output_dimension", &E3TensorProductKokkos::output_dimension)
        .def_property_readonly("weight_size", &E3TensorProductKokkos::weight_size)
        .def_property_readonly(
            "has_internal_weights", &E3TensorProductKokkos::has_internal_weights)
        .def_property_readonly(
            "uses_mh1_fast_path", &E3TensorProductKokkos::uses_mh1_fast_path)
        .def_property_readonly(
            "execution_backend", &E3TensorProductKokkos::execution_backend)
        .def_property_readonly(
            "channel_team_size", &E3TensorProductKokkos::channel_team_size)
        .def_property_readonly(
            "harmonic_team_size", &E3TensorProductKokkos::harmonic_team_size)
        .def("evaluate", [](
            const E3TensorProductKokkos& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights) {
            if(input_1.size()!=static_cast<std::size_t>(self.input_1_dimension())
                ||input_2.size()!=static_cast<std::size_t>(self.input_2_dimension()))
                throw std::invalid_argument(
                    "Kokkos tensor-product input dimensions are inconsistent.");
            const bool use_internal_weights=weights.empty()&&self.has_internal_weights();
            if(!use_internal_weights
                &&weights.size()!=static_cast<std::size_t>(self.weight_size()))
                throw std::invalid_argument(
                    "Kokkos tensor-product weight dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_view;
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_view;
            Kokkos::View<double**,Kokkos::LayoutRight> weights_view;
            set_kokkos_view(input_1_view, input_1, 1, self.input_1_dimension());
            set_kokkos_view(input_2_view, input_2, 1, self.input_2_dimension());
            set_kokkos_view(
                weights_view,weights,1,use_internal_weights?0:self.weight_size());
            Kokkos::View<double**,Kokkos::LayoutRight> output(
                "bound e3 tensor output", 1, self.output_dimension());
            self.evaluate(input_1_view, input_2_view, weights_view, output);
            return view2vector(output);
        })
        .def("reverse", [](
            const E3TensorProductKokkos& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights,
            const std::vector<double>& output_adjoint) {
            if(input_1.size()!=static_cast<std::size_t>(self.input_1_dimension())
                ||input_2.size()!=static_cast<std::size_t>(self.input_2_dimension())
                ||output_adjoint.size()
                    !=static_cast<std::size_t>(self.output_dimension()))
                throw std::invalid_argument(
                    "Kokkos tensor-product reverse dimensions are inconsistent.");
            const bool use_internal_weights=weights.empty()&&self.has_internal_weights();
            if(!use_internal_weights
                &&weights.size()!=static_cast<std::size_t>(self.weight_size()))
                throw std::invalid_argument(
                    "Kokkos tensor-product weight dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_view;
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_view;
            Kokkos::View<double**,Kokkos::LayoutRight> weights_view;
            Kokkos::View<double**,Kokkos::LayoutRight> output_adjoint_view;
            set_kokkos_view(input_1_view, input_1, 1, self.input_1_dimension());
            set_kokkos_view(input_2_view, input_2, 1, self.input_2_dimension());
            set_kokkos_view(
                weights_view,weights,1,use_internal_weights?0:self.weight_size());
            set_kokkos_view(
                output_adjoint_view, output_adjoint, 1, self.output_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_adjoint(
                "bound e3 tensor input 1 adjoint", 1, self.input_1_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_adjoint(
                "bound e3 tensor input 2 adjoint", 1, self.input_2_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> weights_adjoint(
                "bound e3 tensor weights adjoint", 1, self.weight_size());
            self.reverse(
                input_1_view, input_2_view, weights_view, output_adjoint_view,
                input_1_adjoint, input_2_adjoint, weights_adjoint);
            return py::make_tuple(
                view2vector(input_1_adjoint),
                view2vector(input_2_adjoint),
                view2vector(weights_adjoint));
        });

    py::class_<E3ProductBasisKokkos>(module, "E3ProductBasisKokkos")
        .def(py::init([](const std::string& definition) {
            return E3ProductBasisKokkos(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3ProductBasisKokkos::input_dimension)
        .def_property_readonly("output_dimension", &E3ProductBasisKokkos::output_dimension)
        .def_property_readonly("uses_compiled_plan", &E3ProductBasisKokkos::uses_compiled_plan)
        .def_property_readonly(
            "uses_standard_host_plan",
            &E3ProductBasisKokkos::uses_standard_host_plan)
        .def_property_readonly("compiled_term_count", &E3ProductBasisKokkos::compiled_term_count)
        .def_property_readonly("workspace_bytes", &E3ProductBasisKokkos::workspace_bytes)
        .def_property_readonly(
            "feature_major_workspace_bytes",
            &E3ProductBasisKokkos::feature_major_workspace_bytes)
        .def("evaluate_batch", [](
            E3ProductBasisKokkos& self,
            const std::vector<double>& node_features,
            const std::vector<double>& skip_connections,
            const std::vector<int>& elements,
            int samples) {
            if(samples<0
                ||node_features.size()!=static_cast<std::size_t>(samples)*self.input_dimension()
                ||skip_connections.size()!=static_cast<std::size_t>(samples)*self.output_dimension()
                ||elements.size()!=static_cast<std::size_t>(samples))
                throw std::invalid_argument("Kokkos product batch dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input,skip,output;
            Kokkos::View<int*> element_view;
            set_kokkos_view(input,node_features,samples,self.input_dimension());
            set_kokkos_view(skip,skip_connections,samples,self.output_dimension());
            set_kokkos_view(element_view,elements);
            Kokkos::realloc(output,samples,self.output_dimension());
            self.evaluate(input,skip,element_view,output);
            return view2vector(output);
        })
        .def("reverse_batch", [](
            E3ProductBasisKokkos& self,
            const std::vector<double>& node_features,
            const std::vector<int>& elements,
            const std::vector<double>& output_adjoint,
            int samples) {
            if(samples<0
                ||node_features.size()!=static_cast<std::size_t>(samples)*self.input_dimension()
                ||output_adjoint.size()!=static_cast<std::size_t>(samples)*self.output_dimension()
                ||elements.size()!=static_cast<std::size_t>(samples))
                throw std::invalid_argument("Kokkos product batch dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight> input,output_seed,input_adjoint,skip_adjoint;
            Kokkos::View<int*> element_view;
            set_kokkos_view(input,node_features,samples,self.input_dimension());
            set_kokkos_view(output_seed,output_adjoint,samples,self.output_dimension());
            set_kokkos_view(element_view,elements);
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            Kokkos::realloc(skip_adjoint,samples,self.output_dimension());
            self.reverse(input,element_view,output_seed,input_adjoint,skip_adjoint);
            return py::make_tuple(view2vector(input_adjoint),view2vector(skip_adjoint));
        })
        .def("reverse_packed_batch", [](
            E3ProductBasisKokkos& self,
            const std::vector<double>& node_features,
            const std::vector<int>& elements,
            const std::vector<double>& output_adjoint,
            int samples) {
            if(samples<0
                ||node_features.size()
                    !=static_cast<std::size_t>(samples)*self.input_dimension()
                ||output_adjoint.size()
                    !=static_cast<std::size_t>(samples)*self.output_dimension()
                ||elements.size()!=static_cast<std::size_t>(samples))
                throw std::invalid_argument(
                    "Kokkos packed product reverse dimensions are inconsistent.");
            Kokkos::View<double**,Kokkos::LayoutRight>
                input,output_seed,input_adjoint,skip_adjoint;
            Kokkos::View<int*> element_view;
            set_kokkos_view(
                input,node_features,samples,self.input_dimension());
            set_kokkos_view(
                output_seed,output_adjoint,samples,self.output_dimension());
            set_kokkos_view(element_view,elements);
            Kokkos::realloc(input_adjoint,samples,self.input_dimension());
            Kokkos::realloc(skip_adjoint,samples,self.output_dimension());
            self.reverse(
                input,element_view,output_seed,input_adjoint,skip_adjoint,
                true,false,E3ProductBasisKokkos::InputLayout::native);
            auto packed=self.packed_input_adjoint();
            auto host=Kokkos::create_mirror_view_and_copy(
                Kokkos::HostSpace(),packed);
            std::vector<double> result(host.size());
            for(std::size_t sample=0;sample<host.extent(0);++sample)
                for(std::size_t column=0;column<host.extent(1);++column)
                    result[sample*host.extent(1)+column]=host(sample,column);
            return result;
        });
#endif
}
