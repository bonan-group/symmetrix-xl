#include "affine_mlp.hpp"

#include <cmath>
#include <stdexcept>

#include "cblas.hpp"
#include "nlohmann/json.hpp"

namespace {

std::vector<double> tensor_values(const nlohmann::json& tensor, const std::string& name)
{
    if (!tensor.contains("shape") || !tensor.contains("values"))
        throw std::invalid_argument("AffineMLP " + name + " tensor is missing shape or values.");
    return tensor.at("values").get<std::vector<double>>();
}

std::vector<int> tensor_shape(const nlohmann::json& tensor, const std::string& name)
{
    if (!tensor.contains("shape"))
        throw std::invalid_argument("AffineMLP " + name + " tensor is missing shape.");
    return tensor.at("shape").get<std::vector<int>>();
}

double silu(double value)
{
    return value/(1.0+std::exp(-value));
}

double silu_derivative(double value)
{
    const double sigmoid = 1.0/(1.0+std::exp(-value));
    return sigmoid+value*sigmoid*(1.0-sigmoid);
}

}  // namespace

AffineMLP::AffineMLP(const nlohmann::json& definition)
{
    if (!definition.contains("layers") || !definition.at("layers").is_array())
        throw std::invalid_argument("AffineMLP requires a layers array.");

    int previous_output_size = -1;
    for (const auto& definition_layer : definition.at("layers")) {
        const auto type = definition_layer.at("type").get<std::string>();
        Layer layer;
        if (type == "linear") {
            layer.type = Layer::Type::Linear;
            const auto shape = tensor_shape(definition_layer.at("weight"), "linear weight");
            if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
                throw std::invalid_argument("AffineMLP linear weight must be rank two.");
            layer.output_size = shape[0];
            layer.input_size = shape[1];
            layer.weight = tensor_values(definition_layer.at("weight"), "linear weight");
            layer.bias = tensor_values(definition_layer.at("bias"), "linear bias");
            if (static_cast<int>(layer.weight.size()) != layer.output_size*layer.input_size
                || static_cast<int>(layer.bias.size()) != layer.output_size)
                throw std::invalid_argument("AffineMLP linear tensor sizes are inconsistent.");
        } else if (type == "layer_norm") {
            layer.type = Layer::Type::LayerNorm;
            const auto normalized_shape = definition_layer.at("normalized_shape").get<std::vector<int>>();
            if (normalized_shape.size() != 1 || normalized_shape[0] <= 0)
                throw std::invalid_argument("AffineMLP only supports one-dimensional LayerNorm.");
            layer.input_size = normalized_shape[0];
            layer.output_size = layer.input_size;
            layer.eps = definition_layer.at("eps").get<double>();
            layer.weight = tensor_values(definition_layer.at("weight"), "LayerNorm weight");
            layer.bias = tensor_values(definition_layer.at("bias"), "LayerNorm bias");
            if (static_cast<int>(layer.weight.size()) != layer.input_size
                || static_cast<int>(layer.bias.size()) != layer.input_size || !(layer.eps > 0.0))
                throw std::invalid_argument("AffineMLP LayerNorm tensors are inconsistent.");
        } else if (type == "silu") {
            if (previous_output_size < 0)
                throw std::invalid_argument("AffineMLP SiLU cannot be the first layer.");
            layer.type = Layer::Type::SiLU;
            layer.input_size = previous_output_size;
            layer.output_size = previous_output_size;
        } else {
            throw std::invalid_argument("AffineMLP has unsupported layer type " + type + ".");
        }
        if (previous_output_size >= 0 && layer.input_size != previous_output_size)
            throw std::invalid_argument("AffineMLP layer dimensions are inconsistent.");
        previous_output_size = layer.output_size;
        layers.push_back(std::move(layer));
    }
    if (layers.empty())
        throw std::invalid_argument("AffineMLP requires at least one layer.");
}

int AffineMLP::input_size() const
{
    for (const auto& layer : layers)
        if (layer.type != Layer::Type::SiLU)
            return layer.input_size;
    throw std::logic_error("AffineMLP has no sized layer.");
}

int AffineMLP::output_size() const
{
    for (auto it=layers.rbegin(); it != layers.rend(); ++it)
        if (it->type != Layer::Type::SiLU)
            return it->output_size;
    throw std::logic_error("AffineMLP has no sized layer.");
}

bool AffineMLP::supports_conditioned_input(int dynamic_input_size) const
{
    return !layers.empty() && layers.front().type == Layer::Type::Linear
        && dynamic_input_size > 0 && dynamic_input_size < layers.front().input_size;
}

std::vector<double> AffineMLP::first_layer_contribution(
    int input_offset,
    const std::vector<double>& values) const
{
    if (layers.empty() || layers.front().type != Layer::Type::Linear
        || input_offset < 0 || values.empty()
        || input_offset+static_cast<int>(values.size()) > layers.front().input_size)
        throw std::invalid_argument("AffineMLP first-layer contribution dimensions are inconsistent.");
    const auto& first = layers.front();
    std::vector<double> result(first.output_size, 0.0);
    for (int row=0; row<first.output_size; ++row)
        for (int column=0; column<static_cast<int>(values.size()); ++column)
            result[row] += first.weight[row*first.input_size+input_offset+column]
                *values[column];
    return result;
}

std::vector<double> AffineMLP::evaluate(const std::vector<double>& input) const
{
    return evaluate_impl(input, nullptr, nullptr, nullptr);
}

std::vector<double> AffineMLP::evaluate_conditioned(
    const std::vector<double>& input,
    const std::vector<double>& first_contribution,
    const std::vector<double>& second_contribution) const
{
    return evaluate_impl(input, &first_contribution, &second_contribution, nullptr);
}

void AffineMLP::evaluate_conditioned_with_directional_derivative(
    const std::vector<double>& input,
    const std::vector<double>& input_derivative,
    const std::vector<double>& first_contribution,
    const std::vector<double>& second_contribution,
    std::vector<double>& output,
    std::vector<double>& output_derivative) const
{
    if (input_derivative.size() != input.size()
        || !supports_conditioned_input(static_cast<int>(input.size())))
        throw std::invalid_argument(
            "AffineMLP directional derivative input dimensions are inconsistent.");
    const int first_width = layers.front().output_size;
    if (static_cast<int>(first_contribution.size()) != first_width
        || static_cast<int>(second_contribution.size()) != first_width)
        throw std::invalid_argument(
            "AffineMLP directional derivative contributions are inconsistent.");

    output = input;
    output_derivative = input_derivative;
    for (int layer_index=0; layer_index<static_cast<int>(layers.size()); ++layer_index) {
        const auto& layer = layers[layer_index];
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = layer_index == 0
                ? static_cast<int>(output.size()) : layer.input_size;
            auto next = layer.bias;
            auto next_derivative = std::vector<double>(layer.output_size, 0.0);
            if (layer_index == 0)
                for (int row=0; row<layer.output_size; ++row)
                    next[row] += first_contribution[row]+second_contribution[row];
            for (int row=0; row<layer.output_size; ++row) {
                for (int column=0; column<active_input_size; ++column) {
                    const double weight = layer.weight[row*layer.input_size+column];
                    next[row] += weight*output[column];
                    next_derivative[row] += weight*output_derivative[column];
                }
            }
            output = std::move(next);
            output_derivative = std::move(next_derivative);
        } else if (layer.type == Layer::Type::LayerNorm) {
            const int width = layer.input_size;
            double mean = 0.0;
            double derivative_mean = 0.0;
            for (int index=0; index<width; ++index) {
                mean += output[index];
                derivative_mean += output_derivative[index];
            }
            mean /= width;
            derivative_mean /= width;
            double variance = 0.0;
            double variance_derivative = 0.0;
            for (int index=0; index<width; ++index) {
                const double centered = output[index]-mean;
                variance += centered*centered;
                variance_derivative += 2.0*centered
                    *(output_derivative[index]-derivative_mean);
            }
            variance /= width;
            variance_derivative /= width;
            const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
            const double inverse_stddev_derivative = -0.5*variance_derivative
                *inverse_stddev*inverse_stddev*inverse_stddev;
            for (int index=0; index<width; ++index) {
                const double centered = output[index]-mean;
                output[index] = centered*inverse_stddev*layer.weight[index]
                    +layer.bias[index];
                output_derivative[index] = layer.weight[index]
                    *((output_derivative[index]-derivative_mean)*inverse_stddev
                        +centered*inverse_stddev_derivative);
            }
        } else {
            for (int index=0; index<static_cast<int>(output.size()); ++index) {
                output_derivative[index] *= silu_derivative(output[index]);
                output[index] = silu(output[index]);
            }
        }
    }
}

std::vector<double> AffineMLP::evaluate_impl(
    const std::vector<double>& input,
    const std::vector<double>* first_contribution,
    const std::vector<double>* second_contribution,
    std::vector<std::vector<double>>* tape) const
{
    const bool conditioned = first_contribution != nullptr;
    if ((!conditioned && static_cast<int>(input.size()) != input_size())
        || (conditioned && !supports_conditioned_input(input.size())))
        throw std::invalid_argument("AffineMLP input size does not match the first layer.");
    if (conditioned) {
        const int width = layers.front().output_size;
        if (static_cast<int>(first_contribution->size()) != width
            || second_contribution == nullptr
            || static_cast<int>(second_contribution->size()) != width)
            throw std::invalid_argument("AffineMLP conditioned contributions have invalid dimensions.");
    }
    auto values = input;
    if (tape) {
        tape->clear();
        tape->reserve(layers.size()+1);
        tape->push_back(values);
    }
    for (int layer_index=0; layer_index<static_cast<int>(layers.size()); ++layer_index) {
        const auto& layer = layers[layer_index];
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = conditioned && layer_index == 0
                ? static_cast<int>(values.size()) : layer.input_size;
            if (static_cast<int>(values.size()) != active_input_size)
                throw std::logic_error("AffineMLP linear input size is inconsistent.");
            auto output = layer.bias;
            if (conditioned && layer_index == 0)
                for (int row=0; row<layer.output_size; ++row)
                    output[row] += (*first_contribution)[row]+(*second_contribution)[row];
            for (int row=0; row<layer.output_size; ++row)
                for (int column=0; column<active_input_size; ++column)
                    output[row] += layer.weight[row*layer.input_size+column]*values[column];
            values = std::move(output);
        } else if (layer.type == Layer::Type::LayerNorm) {
            double mean = 0.0;
            for (const auto value : values)
                mean += value;
            mean /= values.size();
            double variance = 0.0;
            for (const auto value : values)
                variance += (value-mean)*(value-mean);
            variance /= values.size();
            const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
            for (int index=0; index<layer.output_size; ++index)
                values[index] = (values[index]-mean)*inverse_stddev*layer.weight[index]+layer.bias[index];
        } else {
            for (auto& value : values)
                value = silu(value);
        }
        if (tape) tape->push_back(values);
    }
    return values;
}

std::vector<double> AffineMLP::evaluate_gradient(
    const std::vector<double>& input,
    const std::vector<double>& output_adjoint) const
{
    return evaluate_gradient_impl(input, nullptr, nullptr, output_adjoint);
}

std::vector<double> AffineMLP::evaluate_gradient_conditioned(
    const std::vector<double>& input,
    const std::vector<double>& first_contribution,
    const std::vector<double>& second_contribution,
    const std::vector<double>& output_adjoint) const
{
    return evaluate_gradient_impl(
        input, &first_contribution, &second_contribution, output_adjoint);
}

std::vector<double> AffineMLP::evaluate_gradient_impl(
    const std::vector<double>& input,
    const std::vector<double>* first_contribution,
    const std::vector<double>* second_contribution,
    const std::vector<double>& output_adjoint) const
{
    if (static_cast<int>(output_adjoint.size()) != output_size())
        throw std::invalid_argument("AffineMLP gradient inputs have invalid dimensions.");
    std::vector<std::vector<double>> values;
    evaluate_impl(input, first_contribution, second_contribution, &values);

    auto adjoint = output_adjoint;
    for (int layer_index=static_cast<int>(layers.size())-1; layer_index>=0; --layer_index) {
        const auto& layer = layers[layer_index];
        const auto& layer_input = values[layer_index];
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = layer_index == 0 && first_contribution != nullptr
                ? static_cast<int>(input.size()) : layer.input_size;
            auto input_adjoint = std::vector<double>(active_input_size, 0.0);
            for (int row=0; row<layer.output_size; ++row)
                for (int column=0; column<active_input_size; ++column)
                    input_adjoint[column] += layer.weight[row*layer.input_size+column]*adjoint[row];
            adjoint = std::move(input_adjoint);
        } else if (layer.type == Layer::Type::LayerNorm) {
            const int width = layer.input_size;
            double mean = 0.0;
            for (const auto value : layer_input)
                mean += value;
            mean /= width;
            double variance = 0.0;
            for (const auto value : layer_input)
                variance += (value-mean)*(value-mean);
            variance /= width;
            const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
            auto normalized = std::vector<double>(width);
            auto scaled_adjoint = std::vector<double>(width);
            double sum_scaled = 0.0;
            double sum_scaled_normalized = 0.0;
            for (int index=0; index<width; ++index) {
                normalized[index] = (layer_input[index]-mean)*inverse_stddev;
                scaled_adjoint[index] = adjoint[index]*layer.weight[index];
                sum_scaled += scaled_adjoint[index];
                sum_scaled_normalized += scaled_adjoint[index]*normalized[index];
            }
            for (int index=0; index<width; ++index)
                adjoint[index] = inverse_stddev*(
                    width*scaled_adjoint[index]-sum_scaled-normalized[index]*sum_scaled_normalized
                )/width;
        } else {
            for (int index=0; index<static_cast<int>(adjoint.size()); ++index)
                adjoint[index] *= silu_derivative(layer_input[index]);
        }
    }
    return adjoint;
}

const std::vector<double>& AffineMLP::evaluate_conditioned_batch(
    const std::vector<double>& input,
    int samples,
    int dynamic_input_size,
    const std::vector<double>& row_contributions,
    AffineMLPBatchTape& tape) const
{
    if (samples < 0 || !supports_conditioned_input(dynamic_input_size)
        || input.size() != static_cast<std::size_t>(samples)*dynamic_input_size)
        throw std::invalid_argument("AffineMLP conditioned batch input dimensions are inconsistent.");
    const int conditioning_width = layers.front().output_size;
    if (row_contributions.size()
        != static_cast<std::size_t>(samples)*conditioning_width)
        throw std::invalid_argument("AffineMLP conditioned batch contributions have invalid dimensions.");

    tape.samples = samples;
    tape.dynamic_input_size = dynamic_input_size;
    tape.values.resize(layers.size()+1);
    tape.values.front() = input;
    for (int layer_index=0; layer_index<static_cast<int>(layers.size()); ++layer_index) {
        const auto& layer = layers[layer_index];
        const auto& source = tape.values[layer_index];
        auto& destination = tape.values[layer_index+1];
        destination.resize(static_cast<std::size_t>(samples)*layer.output_size);
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = layer_index == 0
                ? dynamic_input_size : layer.input_size;
            if (samples > 0)
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    samples, layer.output_size, active_input_size,
                    1.0,
                    source.data(), active_input_size,
                    layer.weight.data(), layer.input_size,
                    0.0,
                    destination.data(), layer.output_size);
            for (int sample=0; sample<samples; ++sample)
                for (int column=0; column<layer.output_size; ++column) {
                    const std::size_t offset =
                        static_cast<std::size_t>(sample)*layer.output_size+column;
                    destination[offset] += layer.bias[column];
                    if (layer_index == 0)
                        destination[offset] += row_contributions[offset];
                }
        } else if (layer.type == Layer::Type::LayerNorm) {
            for (int sample=0; sample<samples; ++sample) {
                const double* values = source.data()
                    +static_cast<std::size_t>(sample)*layer.input_size;
                double mean = 0.0;
                for (int column=0; column<layer.input_size; ++column)
                    mean += values[column];
                mean /= layer.input_size;
                double variance = 0.0;
                for (int column=0; column<layer.input_size; ++column)
                    variance += (values[column]-mean)*(values[column]-mean);
                variance /= layer.input_size;
                const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
                for (int column=0; column<layer.input_size; ++column)
                    destination[static_cast<std::size_t>(sample)*layer.input_size+column]
                        = (values[column]-mean)*inverse_stddev*layer.weight[column]
                        + layer.bias[column];
            }
        } else {
            for (std::size_t index=0; index<source.size(); ++index)
                destination[index] = silu(source[index]);
        }
    }
    return tape.values.back();
}

void AffineMLP::reverse_conditioned_batch(
    const std::vector<double>& output_adjoint,
    const AffineMLPBatchTape& tape,
    std::vector<double>& input_adjoint,
    AffineMLPBatchWorkspace& workspace) const
{
    if (tape.samples < 0 || !supports_conditioned_input(tape.dynamic_input_size)
        || tape.values.size() != layers.size()+1
        || output_adjoint.size()
            != static_cast<std::size_t>(tape.samples)*output_size())
        throw std::invalid_argument("AffineMLP conditioned batch tape dimensions are inconsistent.");
    workspace.adjoint = output_adjoint;
    for (int layer_index=static_cast<int>(layers.size())-1; layer_index>=0; --layer_index) {
        const auto& layer = layers[layer_index];
        const auto& layer_input = tape.values[layer_index];
        const int active_input_size = layer_index == 0
            ? tape.dynamic_input_size : layer.input_size;
        if (layer_input.size()
            != static_cast<std::size_t>(tape.samples)*active_input_size)
            throw std::invalid_argument("AffineMLP conditioned batch tape dimensions are inconsistent.");
        workspace.scratch.resize(
            static_cast<std::size_t>(tape.samples)*active_input_size);
        if (layer.type == Layer::Type::Linear) {
            if (tape.samples > 0)
                cblas_dgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    tape.samples, active_input_size, layer.output_size,
                    1.0,
                    workspace.adjoint.data(), layer.output_size,
                    layer.weight.data(), layer.input_size,
                    0.0,
                    workspace.scratch.data(), active_input_size);
        } else if (layer.type == Layer::Type::LayerNorm) {
            for (int sample=0; sample<tape.samples; ++sample) {
                const double* values = layer_input.data()
                    +static_cast<std::size_t>(sample)*layer.input_size;
                const double* adjoint = workspace.adjoint.data()
                    +static_cast<std::size_t>(sample)*layer.input_size;
                double mean = 0.0;
                for (int column=0; column<layer.input_size; ++column)
                    mean += values[column];
                mean /= layer.input_size;
                double variance = 0.0;
                for (int column=0; column<layer.input_size; ++column)
                    variance += (values[column]-mean)*(values[column]-mean);
                variance /= layer.input_size;
                const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
                double sum_scaled = 0.0;
                double sum_scaled_normalized = 0.0;
                for (int column=0; column<layer.input_size; ++column) {
                    const double normalized = (values[column]-mean)*inverse_stddev;
                    const double scaled = adjoint[column]*layer.weight[column];
                    sum_scaled += scaled;
                    sum_scaled_normalized += scaled*normalized;
                }
                for (int column=0; column<layer.input_size; ++column) {
                    const double normalized = (values[column]-mean)*inverse_stddev;
                    const double scaled = adjoint[column]*layer.weight[column];
                    workspace.scratch[
                        static_cast<std::size_t>(sample)*layer.input_size+column]
                        = inverse_stddev*(
                            layer.input_size*scaled-sum_scaled
                            -normalized*sum_scaled_normalized)/layer.input_size;
                }
            }
        } else {
            for (std::size_t index=0; index<layer_input.size(); ++index)
                workspace.scratch[index] = workspace.adjoint[index]
                    *silu_derivative(layer_input[index]);
        }
        workspace.adjoint.swap(workspace.scratch);
    }
    input_adjoint = workspace.adjoint;
}
