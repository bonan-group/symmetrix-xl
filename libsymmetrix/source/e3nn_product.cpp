#include "e3nn_product.hpp"

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>

namespace {

E3ProductBasis::Tensor tensor_from_json(const nlohmann::json& value)
{
    E3ProductBasis::Tensor tensor;
    tensor.shape = value.at("shape").get<std::vector<int>>();
    tensor.values = value.at("values").get<std::vector<double>>();
    std::size_t size = 1;
    for (int dimension : tensor.shape) {
        if (dimension <= 0
            || size > std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(dimension))
            throw std::invalid_argument("MACE_Nonlinear product tensor has an invalid shape.");
        size *= static_cast<std::size_t>(dimension);
    }
    if (size != tensor.values.size())
        throw std::invalid_argument("MACE_Nonlinear product tensor has an invalid shape.");
    return tensor;
}

} // namespace

namespace {

struct MonomialKey {
    int degree = 0;
    std::array<int,3> indices{};

    bool operator<(const MonomialKey& other) const
    {
        if (degree != other.degree) return degree < other.degree;
        return indices < other.indices;
    }
};

} // namespace

int E3ProductBasis::Tensor::offset(const std::vector<int>& index) const
{
    if (index.size() != shape.size())
        throw std::invalid_argument("MACE_Nonlinear product tensor rank mismatch.");
    int result = 0;
    for (int axis=0; axis<static_cast<int>(shape.size()); ++axis) {
        if (index[axis] < 0 || index[axis] >= shape[axis])
            throw std::out_of_range("MACE_Nonlinear product tensor index is out of range.");
        result = result * shape[axis] + index[axis];
    }
    return result;
}

E3ProductBasis::E3ProductBasis(const nlohmann::json& data)
    : input(data.at("symmetric_contractions").at("irreps_in").get<std::string>()),
      output(data.at("symmetric_contractions").at("irreps_out").get<std::string>()),
      linear(data.at("linear")),
      use_sc(data.at("use_sc").get<bool>()),
      num_features(input.blocks.front().multiplicity),
      angular_dimension(0)
{
    if (linear.input_dimension() != output.dimension() || linear.output_dimension() != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear product linear has incompatible irreps.");
    for (const auto& block : input.blocks) {
        if (block.multiplicity != num_features)
            throw std::invalid_argument("MACE_Nonlinear product requires a common feature multiplicity.");
        angular_dimension += 2*block.l + 1;
    }
    for (const auto& block : output.blocks)
        if (block.multiplicity != num_features)
            throw std::invalid_argument("MACE_Nonlinear product output multiplicity is incompatible.");
    const auto& serialized = data.at("symmetric_contractions").at("contractions");
    if (serialized.size() != output.blocks.size())
        throw std::invalid_argument("MACE_Nonlinear product has incompatible contraction outputs.");
    for (int contraction_index=0;
         contraction_index<static_cast<int>(serialized.size()); ++contraction_index) {
        const auto& value = serialized.at(contraction_index);
        const auto& output_block = output.blocks[contraction_index];
        Contraction contraction;
        contraction.correlation = value.at("correlation").get<int>();
        contraction.weights_max = tensor_from_json(value.at("weights_max"));
        for (const auto& weight : value.at("weights"))
            contraction.weights.push_back(tensor_from_json(weight));
        for (const auto& u : value.at("u_tensors"))
            contraction.u_tensors.push_back(tensor_from_json(u));
        if (contraction.correlation < 1
            || static_cast<int>(contraction.u_tensors.size()) != contraction.correlation
            || static_cast<int>(contraction.weights.size()) != contraction.correlation-1)
            throw std::invalid_argument("MACE_Nonlinear product has invalid correlation data.");
        int num_elements = -1;
        for (int degree=1; degree<=contraction.correlation; ++degree) {
            const auto& u = contraction.u_tensors[degree-1];
            const auto& weights = degree == contraction.correlation
                ? contraction.weights_max
                : contraction.weights[contraction.correlation-degree-1];
            const int output_axes = output_block.l == 0 ? 0 : 1;
            if (static_cast<int>(u.shape.size()) != degree+output_axes+1
                || (output_axes && u.shape[0] != 2*output_block.l+1)
                || weights.shape.size() != 3
                || weights.shape[1] != u.shape.back()
                || weights.shape[2] != num_features)
                throw std::invalid_argument("Unsupported MACE_Nonlinear product tensor layout.");
            for (int axis=0; axis<degree; ++axis)
                if (u.shape[output_axes+axis] != angular_dimension)
                    throw std::invalid_argument("Unsupported MACE_Nonlinear product angular tensor layout.");
            if (num_elements < 0)
                num_elements = weights.shape[0];
            else if (weights.shape[0] != num_elements)
                throw std::invalid_argument("MACE_Nonlinear product element dimensions are inconsistent.");
        }
        contractions.push_back(std::move(contraction));
    }
    if (has_mh1_product_layout(data)) compile_mh1_product();
}

int E3ProductBasis::compiled_term_count() const
{
    int result = 0;
    for (const auto& block : compiled_blocks)
        result += static_cast<int>(block.terms.size());
    return result;
}

bool E3ProductBasis::has_mh1_product_layout(const nlohmann::json& data) const
{
    if (!use_sc || !data.value("use_agnostic_product", false)
        || num_features <= 0
        || (input.blocks.size() != 3 && input.blocks.size() != 4))
        return false;
    for (int index=0; index<static_cast<int>(input.blocks.size()); ++index) {
        const auto& block = input.blocks[index];
        if (block.multiplicity != num_features || block.l != index
            || block.parity != (index%2 == 0 ? 1 : -1))
            return false;
    }
    const bool first_product = output.blocks.size() == 2
        && output.blocks[0].multiplicity == num_features && output.blocks[0].l == 0
        && output.blocks[0].parity == 1
        && output.blocks[1].multiplicity == num_features && output.blocks[1].l == 1
        && output.blocks[1].parity == -1;
    const bool second_product = output.blocks.size() == 1
        && output.blocks[0].multiplicity == num_features && output.blocks[0].l == 0
        && output.blocks[0].parity == 1;
    if (!first_product && !second_product) return false;
    for (const auto& contraction : contractions)
        if (contraction.correlation != 3 || contraction.weights_max.shape[0] != 1)
            return false;
    return true;
}

void E3ProductBasis::compile_mh1_product()
{
    compiled_blocks.clear();
    int angular_offset = 0;
    for (int block_index=0; block_index<static_cast<int>(output.blocks.size()); ++block_index) {
        const auto& output_block = output.blocks[block_index];
        const auto& contraction = contractions[block_index];
        CompiledBlock compiled;
        compiled.angular_offset = angular_offset;
        compiled.width = 2*output_block.l+1;
        compiled.num_elements = contraction.weights_max.shape[0];
        compiled.component_offsets.push_back(0);

        for (int component=0; component<compiled.width; ++component) {
            std::map<MonomialKey,std::vector<double>> coefficients;
            for (int degree=1; degree<=contraction.correlation; ++degree) {
                const auto& u = contraction.u_tensors[degree-1];
                const auto& weights = degree == contraction.correlation
                    ? contraction.weights_max
                    : contraction.weights[contraction.correlation-degree-1];
                const int parameters = u.shape.back();
                int tuples = 1;
                for (int axis=0; axis<degree; ++axis) tuples *= angular_dimension;
                const int component_offset = output_block.l == 0
                    ? 0 : component*tuples*parameters;

                for (int tuple=0; tuple<tuples; ++tuple) {
                    MonomialKey key;
                    key.degree = degree;
                    int remainder = tuple;
                    for (int axis=degree-1; axis>=0; --axis) {
                        key.indices[axis] = remainder%angular_dimension;
                        remainder /= angular_dimension;
                    }
                    std::sort(key.indices.begin(), key.indices.begin()+degree);
                    for (int parameter=0; parameter<parameters; ++parameter) {
                        const double u_value =
                            u.values[component_offset+tuple*parameters+parameter];
                        if (u_value == 0.0) continue;
                        auto& values = coefficients[key];
                        if (values.empty())
                            values.assign(compiled.num_elements*num_features, 0.0);
                        for (int element=0; element<compiled.num_elements; ++element) {
                            const int weight_offset =
                                (element*parameters+parameter)*num_features;
                            const int coefficient_offset = element*num_features;
                            for (int feature=0; feature<num_features; ++feature)
                                values[coefficient_offset+feature] += u_value
                                    *weights.values[weight_offset+feature];
                        }
                    }
                }
            }
            for (auto& [key, values] : coefficients)
                compiled.terms.push_back({key.degree, key.indices, std::move(values)});
            compiled.component_offsets.push_back(static_cast<int>(compiled.terms.size()));
        }
        angular_offset += compiled.width;
        compiled_blocks.push_back(std::move(compiled));
    }
}

void E3ProductBasis::evaluate_compiled(
    const std::vector<double>& feature_major,
    int element,
    std::vector<double>& product_major) const
{
    for (const auto& block : compiled_blocks) {
        if (element < 0 || element >= block.num_elements)
            throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
        for (int component=0; component<block.width; ++component) {
            const int first = block.component_offsets[component];
            const int last = block.component_offsets[component+1];
            for (int term_index=first; term_index<last; ++term_index) {
                const auto& term = block.terms[term_index];
                const double* coefficients =
                    term.coefficients.data()+element*num_features;
                for (int feature=0; feature<num_features; ++feature) {
                    const double* values =
                        feature_major.data()+feature*angular_dimension;
                    double monomial = values[term.indices[0]];
                    if (term.degree > 1) monomial *= values[term.indices[1]];
                    if (term.degree > 2) monomial *= values[term.indices[2]];
                    product_major[feature*angular_dimension
                                  +block.angular_offset+component]
                        += coefficients[feature]*monomial;
                }
            }
        }
    }
}

void E3ProductBasis::reverse_compiled(
    const std::vector<double>& feature_major,
    int element,
    std::span<const double> contracted_adjoint,
    std::vector<double>& feature_major_adjoint) const
{
    for (int block_index=0; block_index<static_cast<int>(compiled_blocks.size()); ++block_index) {
        const auto& block = compiled_blocks[block_index];
        const auto& output_block = output.blocks[block_index];
        if (element < 0 || element >= block.num_elements)
            throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
        for (int component=0; component<block.width; ++component) {
            const int first = block.component_offsets[component];
            const int last = block.component_offsets[component+1];
            for (int term_index=first; term_index<last; ++term_index) {
                const auto& term = block.terms[term_index];
                const double* coefficients =
                    term.coefficients.data()+element*num_features;
                for (int feature=0; feature<num_features; ++feature) {
                    const double common = coefficients[feature]
                        *contracted_adjoint[output_block.offset
                            +feature*block.width+component];
                    const int base = feature*angular_dimension;
                    const int i0 = term.indices[0];
                    if (term.degree == 1) {
                        feature_major_adjoint[base+i0] += common;
                        continue;
                    }
                    const int i1 = term.indices[1];
                    const double x0 = feature_major[base+i0];
                    const double x1 = feature_major[base+i1];
                    if (term.degree == 2) {
                        feature_major_adjoint[base+i0] += common*x1;
                        feature_major_adjoint[base+i1] += common*x0;
                        continue;
                    }
                    const int i2 = term.indices[2];
                    const double x2 = feature_major[base+i2];
                    feature_major_adjoint[base+i0] += common*x1*x2;
                    feature_major_adjoint[base+i1] += common*x0*x2;
                    feature_major_adjoint[base+i2] += common*x0*x1;
                }
            }
        }
    }
}

void E3ProductBasis::make_feature_major(
    std::span<const double> node_features,
    std::vector<double>& feature_major) const
{
    if (static_cast<int>(node_features.size()) != input.dimension())
        throw std::invalid_argument("MACE_Nonlinear product feature size is invalid.");
    feature_major.resize(num_features*angular_dimension);
    int angular_offset = 0;
    for (const auto& block : input.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                feature_major[feature*angular_dimension+angular_offset+component]
                    = node_features[block.offset+feature*width+component];
        angular_offset += width;
    }
}

void E3ProductBasis::make_irrep_major(
    const std::vector<double>& feature_major,
    std::span<double> result) const
{
    if (static_cast<int>(result.size()) != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear product contracted size is invalid.");
    std::fill(result.begin(), result.end(), 0.0);
    int angular_offset = 0;
    for (const auto& block : output.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                result[block.offset+feature*width+component]
                    = feature_major[feature*angular_dimension+angular_offset+component];
        angular_offset += width;
    }
}

double E3ProductBasis::evaluate_term(
    const Tensor& u,
    const Tensor& weights,
    const std::vector<double>& feature_major,
    int output_component,
    int feature,
    int element) const
{
    const int output_axes = output_component >= 0 ? 1 : 0;
    const int degree = static_cast<int>(u.shape.size()) - output_axes - 1;
    if (degree < 1 || u.shape.back() != weights.shape[1]
        || u.shape[output_axes] != angular_dimension
        || weights.shape.size() != 3 || feature >= weights.shape[2])
        throw std::invalid_argument("Unsupported MACE_Nonlinear product tensor layout.");
    const int parameters = u.shape.back();
    int tuples = 1;
    for (int axis=0; axis<degree; ++axis) tuples *= u.shape[output_axes+axis];
    const int u_component_offset = output_component < 0 ? 0 : output_component*tuples*parameters;
    const int weight_element_offset = element*parameters*num_features;
    double sum = 0.0;
    for (int tuple=0; tuple<tuples; ++tuple) {
        int remainder = tuple;
        double monomial = 1.0;
        for (int axis=degree-1; axis>=0; --axis) {
            const int dimension = u.shape[output_axes+axis];
            const int index = remainder%dimension;
            remainder /= dimension;
            monomial *= feature_major[feature*angular_dimension+index];
        }
        double coefficient = 0.0;
        const int u_offset = u_component_offset+tuple*parameters;
        for (int parameter=0; parameter<parameters; ++parameter)
            coefficient += u.values[u_offset+parameter]
                * weights.values[weight_element_offset+parameter*num_features+feature];
        sum += coefficient*monomial;
    }
    return sum;
}

void E3ProductBasis::reverse_term(
    const Tensor& u,
    const Tensor& weights,
    const std::vector<double>& feature_major,
    int output_component,
    int feature,
    int element,
    double output_adjoint,
    std::vector<double>& feature_major_adjoint) const
{
    const int output_axes = output_component >= 0 ? 1 : 0;
    const int degree = static_cast<int>(u.shape.size()) - output_axes - 1;
    const int parameters = u.shape.back();
    int tuples = 1;
    for (int axis=0; axis<degree; ++axis) tuples *= u.shape[output_axes+axis];
    const int u_component_offset = output_component < 0 ? 0 : output_component*tuples*parameters;
    const int weight_element_offset = element*parameters*num_features;
    std::vector<int> indices(degree);
    for (int tuple=0; tuple<tuples; ++tuple) {
        int remainder = tuple;
        for (int axis=degree-1; axis>=0; --axis) {
            const int dimension = u.shape[output_axes+axis];
            indices[axis] = remainder%dimension;
            remainder /= dimension;
        }
        double coefficient = 0.0;
        const int u_offset = u_component_offset+tuple*parameters;
        for (int parameter=0; parameter<parameters; ++parameter)
            coefficient += u.values[u_offset+parameter]
                * weights.values[weight_element_offset+parameter*num_features+feature];
        coefficient *= output_adjoint;
        for (int differentiated=0; differentiated<degree; ++differentiated) {
            double derivative = coefficient;
            for (int factor=0; factor<degree; ++factor)
                if (factor != differentiated)
                    derivative *= feature_major[feature*angular_dimension+indices[factor]];
            feature_major_adjoint[feature*angular_dimension+indices[differentiated]] += derivative;
        }
    }
}

std::vector<double> E3ProductBasis::evaluate(
    const std::vector<double>& node_features,
    const std::vector<double>& skip_connection,
    int element) const
{
    std::vector<double> contracted(linear.input_dimension());
    std::vector<double> feature_major;
    std::vector<double> product_major;
    evaluate_contraction(
        node_features, element, contracted, feature_major, product_major);
    auto result = linear.evaluate(contracted);
    if (use_sc) {
        if (skip_connection.size() != result.size())
            throw std::invalid_argument("MACE_Nonlinear product skip connection size is invalid.");
        for (int i=0; i<static_cast<int>(result.size()); ++i)
            result[i] += skip_connection[i];
    }
    return result;
}

void E3ProductBasis::evaluate_contraction(
    std::span<const double> node_features,
    int element,
    std::span<double> contracted,
    std::vector<double>& feature_major,
    std::vector<double>& product_major) const
{
    make_feature_major(node_features, feature_major);
    product_major.assign(num_features*angular_dimension, 0.0);
    if (!compiled_blocks.empty()) {
        evaluate_compiled(feature_major, element, product_major);
    } else {
        int output_angular_offset = 0;
        for (int block_index=0; block_index<static_cast<int>(output.blocks.size()); ++block_index) {
            const auto& block = output.blocks[block_index];
            const auto& contraction = contractions[block_index];
            if (element < 0 || element >= contraction.weights_max.shape[0])
                throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
            for (int feature=0; feature<num_features; ++feature)
                for (int component=0; component<2*block.l+1; ++component)
                    for (int degree=1; degree<=contraction.correlation; ++degree) {
                        const auto& u = contraction.u_tensors[degree-1];
                        const auto& weights = degree == contraction.correlation
                            ? contraction.weights_max
                            : contraction.weights[contraction.correlation-degree-1];
                        product_major[feature*angular_dimension+output_angular_offset+component]
                            += evaluate_term(u, weights, feature_major,
                                             block.l == 0 ? -1 : component, feature, element);
                    }
            output_angular_offset += 2*block.l+1;
        }
    }
    make_irrep_major(product_major, contracted);
}

void E3ProductBasis::reverse(
    const std::vector<double>& node_features,
    int element,
    const std::vector<double>& output_adjoint,
    std::vector<double>& node_features_adjoint,
    std::vector<double>& skip_connection_adjoint) const
{
    if (static_cast<int>(output_adjoint.size()) != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear product output adjoint size is invalid.");
    std::vector<double> contracted_adjoint;
    linear.reverse(output_adjoint, contracted_adjoint);
    node_features_adjoint.assign(input.dimension(), 0.0);
    std::vector<double> feature_major;
    std::vector<double> feature_major_adjoint;
    reverse_contraction(
        node_features, element, contracted_adjoint, node_features_adjoint,
        feature_major, feature_major_adjoint);
    skip_connection_adjoint = use_sc
        ? output_adjoint
        : std::vector<double>(output.dimension(), 0.0);
}

void E3ProductBasis::reverse_contraction(
    std::span<const double> node_features,
    int element,
    std::span<const double> contracted_adjoint,
    std::span<double> node_features_adjoint,
    std::vector<double>& feature_major,
    std::vector<double>& feature_major_adjoint) const
{
    if (static_cast<int>(contracted_adjoint.size()) != linear.input_dimension())
        throw std::invalid_argument("MACE_Nonlinear product contracted adjoint size is invalid.");
    if (static_cast<int>(node_features_adjoint.size()) != input.dimension())
        throw std::invalid_argument("MACE_Nonlinear product feature adjoint size is invalid.");
    make_feature_major(node_features, feature_major);
    feature_major_adjoint.assign(num_features*angular_dimension, 0.0);
    if (!compiled_blocks.empty()) {
        reverse_compiled(
            feature_major, element, contracted_adjoint, feature_major_adjoint);
    } else {
        for (int block_index=0; block_index<static_cast<int>(output.blocks.size()); ++block_index) {
            const auto& block = output.blocks[block_index];
            const auto& contraction = contractions[block_index];
            const int width = 2*block.l+1;
            for (int feature=0; feature<num_features; ++feature)
                for (int component=0; component<width; ++component) {
                    const double adjoint = contracted_adjoint[block.offset+feature*width+component];
                    for (int degree=1; degree<=contraction.correlation; ++degree) {
                        const auto& weights = degree == contraction.correlation
                            ? contraction.weights_max
                            : contraction.weights[contraction.correlation-degree-1];
                        reverse_term(contraction.u_tensors[degree-1], weights, feature_major,
                                     block.l == 0 ? -1 : component, feature, element,
                                     adjoint, feature_major_adjoint);
                    }
                }
        }
    }
    std::fill(node_features_adjoint.begin(), node_features_adjoint.end(), 0.0);
    int angular_offset = 0;
    for (const auto& block : input.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                node_features_adjoint[block.offset+feature*width+component]
                    = feature_major_adjoint[
                        feature*angular_dimension+angular_offset+component];
        angular_offset += width;
    }
}

void E3ProductBasis::evaluate_batch(
    const std::vector<double>& node_features,
    const std::vector<double>& skip_connections,
    const std::vector<int>& elements,
    int samples,
    std::vector<double>& output_values,
    E3ProductBasisBatchWorkspace& workspace) const
{
    if (samples < 0
        || node_features.size() != static_cast<std::size_t>(samples)*input.dimension()
        || elements.size() != static_cast<std::size_t>(samples)
        || (use_sc
            && skip_connections.size() != static_cast<std::size_t>(samples)*output.dimension()))
        throw std::invalid_argument("MACE_Nonlinear product batch input size is invalid.");
    workspace.contracted.resize(static_cast<std::size_t>(samples)*linear.input_dimension());
    for (int sample=0; sample<samples; ++sample) {
        const std::span<const double> node(
            node_features.data()+static_cast<std::size_t>(sample)*input.dimension(),
            input.dimension());
        const std::span<double> contracted(
            workspace.contracted.data()
                +static_cast<std::size_t>(sample)*linear.input_dimension(),
            linear.input_dimension());
        evaluate_contraction(
            node, elements[sample], contracted,
            workspace.feature_major, workspace.product_major);
    }
    linear.evaluate_batch(workspace.contracted, samples, output_values, workspace.linear);
    if (use_sc)
        for (std::size_t index=0; index<output_values.size(); ++index)
            output_values[index] += skip_connections[index];
}

void E3ProductBasis::reverse_batch(
    const std::vector<double>& node_features,
    const std::vector<int>& elements,
    const std::vector<double>& output_adjoint,
    int samples,
    std::vector<double>& node_features_adjoint,
    std::vector<double>& skip_connection_adjoint,
    E3ProductBasisBatchWorkspace& workspace) const
{
    if (samples < 0
        || node_features.size() != static_cast<std::size_t>(samples)*input.dimension()
        || output_adjoint.size() != static_cast<std::size_t>(samples)*output.dimension()
        || elements.size() != static_cast<std::size_t>(samples))
        throw std::invalid_argument("MACE_Nonlinear product batch input size is invalid.");
    workspace.contracted.clear();
    linear.reverse_batch(output_adjoint, samples, workspace.contracted, workspace.linear);
    node_features_adjoint.assign(
        static_cast<std::size_t>(samples)*input.dimension(), 0.0);
    for (int sample=0; sample<samples; ++sample) {
        const std::span<const double> node(
            node_features.data()+static_cast<std::size_t>(sample)*input.dimension(),
            input.dimension());
        const std::span<const double> contracted_adjoint(
            workspace.contracted.data()
                +static_cast<std::size_t>(sample)*linear.input_dimension(),
            linear.input_dimension());
        const std::span<double> node_adjoint(
            node_features_adjoint.data()
                +static_cast<std::size_t>(sample)*input.dimension(),
            input.dimension());
        reverse_contraction(
            node, elements[sample], contracted_adjoint, node_adjoint,
            workspace.feature_major, workspace.feature_adjoint);
    }
    if (use_sc)
        skip_connection_adjoint = output_adjoint;
    else
        skip_connection_adjoint.assign(
            static_cast<std::size_t>(samples)*output.dimension(), 0.0);
}
