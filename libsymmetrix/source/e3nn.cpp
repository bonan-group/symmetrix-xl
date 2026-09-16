#include "e3nn.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "cblas.hpp"

namespace {

std::vector<double> tensor_values(const nlohmann::json& tensor)
{
    return tensor.at("values").get<std::vector<double>>();
}

std::vector<int> tensor_shape(const nlohmann::json& tensor)
{
    return tensor.at("shape").get<std::vector<int>>();
}

int product(const std::vector<int>& shape)
{
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

E3Instruction parse_instruction(const nlohmann::json& value, bool tensor_product)
{
    E3Instruction instruction;
    instruction.output = value.at("i_out").get<int>();
    instruction.path_weight = value.value("path_weight", 1.0);
    instruction.path_shape = value.at("path_shape").get<std::vector<int>>();
    if (tensor_product) {
        instruction.input_1 = value.at("i_in1").get<int>();
        instruction.input_2 = value.at("i_in2").get<int>();
        instruction.mode = value.at("connection_mode").get<std::string>();
        instruction.has_weight = value.at("has_weight").get<bool>();
        if (!value.contains("wigner_3j"))
            throw std::invalid_argument("MACE_Nonlinear tensor-product instruction is missing wigner_3j.");
        instruction.wigner_3j = tensor_values(value.at("wigner_3j"));
        instruction.wigner_shape = tensor_shape(value.at("wigner_3j"));
    } else {
        instruction.input_1 = value.at("i_in").get<int>();
        instruction.mode = "linear";
        instruction.has_weight = true;
    }
    return instruction;
}

void validate_vector_size(const std::vector<double>& value, int expected, const char* name)
{
    if (static_cast<int>(value.size()) != expected)
        throw std::invalid_argument(std::string("Invalid ") + name + " size in MACE_Nonlinear JSON.");
}

void validate_batch_size(
    const std::vector<double>& value, int samples, int width, const char* name)
{
    if (samples < 0 || width < 0
        || static_cast<std::size_t>(samples) > std::numeric_limits<std::size_t>::max()
            / static_cast<std::size_t>(std::max(width, 1))
        || value.size() != static_cast<std::size_t>(samples)*static_cast<std::size_t>(width))
        throw std::invalid_argument(std::string("Invalid ") + name + " size in MACE_Nonlinear JSON.");
}

double sigmoid(double value)
{
    if (value >= 0.0) {
        const double z = std::exp(-value);
        return 1.0 / (1.0 + z);
    }
    const double z = std::exp(value);
    return z / (1.0 + z);
}

} // namespace

Irreps::Irreps(const std::string& specification)
{
    std::size_t start = 0;
    while (start < specification.size()) {
        const std::size_t end = specification.find('+', start);
        std::string token = specification.substr(start, end == std::string::npos ? std::string::npos : end-start);
        token.erase(std::remove_if(token.begin(), token.end(), [](unsigned char c) { return std::isspace(c); }), token.end());
        const std::size_t x = token.find('x');
        const std::size_t parity_position = token.find_first_of("eo", x == std::string::npos ? 0 : x+1);
        if (x == std::string::npos || parity_position == std::string::npos || parity_position + 1 != token.size())
            throw std::invalid_argument("Unsupported irreps specification in MACE_Nonlinear JSON: " + specification);
        const int multiplicity = std::stoi(token.substr(0, x));
        const int l = std::stoi(token.substr(x+1, parity_position-x-1));
        if (multiplicity <= 0 || l < 0)
            throw std::invalid_argument("Invalid irreps specification in MACE_Nonlinear JSON.");
        blocks.push_back({multiplicity, l, token[parity_position] == 'e' ? 1 : -1, dimension_});
        dimension_ += blocks.back().dimension();
        if (end == std::string::npos)
            break;
        start = end + 1;
    }
    if (blocks.empty())
        throw std::invalid_argument("MACE_Nonlinear JSON has an empty irreps specification.");
}

E3Linear::E3Linear(const nlohmann::json& data)
    : input(data.at("irreps_in").get<std::string>()),
      output(data.at("irreps_out").get<std::string>()),
      weights(tensor_values(data.at("weight"))),
      bias(tensor_values(data.at("bias"))),
      output_mask(tensor_values(data.at("output_mask")))
{
    int weight_offset = 0;
    for (const auto& entry : data.at("instructions")) {
        auto instruction = parse_instruction(entry, false);
        if (instruction.input_1 < 0 || instruction.input_1 >= static_cast<int>(input.blocks.size())
            || instruction.output < 0 || instruction.output >= static_cast<int>(output.blocks.size()))
            throw std::invalid_argument("MACE_Nonlinear linear instruction has an invalid irrep index.");
        if (instruction.path_shape.size() != 2)
            throw std::invalid_argument("MACE_Nonlinear linear instruction must have a two-dimensional weight.");
        instruction.weight_offset = weight_offset;
        weight_offset += product(instruction.path_shape);
        instructions.push_back(std::move(instruction));
    }
    validate_vector_size(weights, weight_offset, "linear weight");
    validate_vector_size(output_mask, output.dimension(), "linear output_mask");
    if (!bias.empty() && static_cast<int>(bias.size()) != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear linear bias has an invalid size.");
}

std::vector<double> E3Linear::evaluate(const std::vector<double>& x) const
{
    validate_vector_size(x, input.dimension(), "linear input");
    std::vector<double> result(output.dimension(), 0.0);
    for (const auto& instruction : instructions) {
        const auto& in = input.blocks[instruction.input_1];
        const auto& out = output.blocks[instruction.output];
        if (instruction.path_shape[0] != in.multiplicity || instruction.path_shape[1] != out.multiplicity || in.l != out.l)
            throw std::invalid_argument("Unsupported MACE_Nonlinear e3nn Linear path.");
        const int width = 2*in.l + 1;
        cblas_dgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            out.multiplicity, width, in.multiplicity,
            instruction.path_weight,
            weights.data()+instruction.weight_offset, out.multiplicity,
            x.data()+in.offset, width,
            1.0,
            result.data()+out.offset, width);
    }
    for (int i=0; i<output.dimension(); ++i)
        result[i] = (result[i] + (bias.empty() ? 0.0 : bias[i])) * output_mask[i];
    return result;
}

void E3Linear::reverse(const std::vector<double>& output_adj, std::vector<double>& input_adj) const
{
    validate_vector_size(output_adj, output.dimension(), "linear output adjoint");
    if (static_cast<int>(input_adj.size()) != input.dimension())
        input_adj.assign(input.dimension(), 0.0);
    std::vector<double> masked_output_adjoint(output.dimension());
    for (int index=0; index<output.dimension(); ++index)
        masked_output_adjoint[index] = output_mask[index]*output_adj[index];
    for (const auto& instruction : instructions) {
        const auto& in = input.blocks[instruction.input_1];
        const auto& out = output.blocks[instruction.output];
        const int width = 2*in.l + 1;
        cblas_dgemm(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            in.multiplicity, width, out.multiplicity,
            instruction.path_weight,
            weights.data()+instruction.weight_offset, out.multiplicity,
            masked_output_adjoint.data()+out.offset, width,
            1.0,
            input_adj.data()+in.offset, width);
    }
}

void E3Linear::evaluate_batch(
    const std::vector<double>& input_values,
    int samples,
    std::vector<double>& output_values,
    E3LinearBatchWorkspace& workspace) const
{
    validate_batch_size(input_values, samples, input.dimension(), "linear batch input");
    output_values.assign(static_cast<std::size_t>(samples)*output.dimension(), 0.0);
    for (const auto& instruction : instructions) {
        const auto& in = input.blocks[instruction.input_1];
        const auto& out = output.blocks[instruction.output];
        if (instruction.path_shape[0] != in.multiplicity
            || instruction.path_shape[1] != out.multiplicity || in.l != out.l)
            throw std::invalid_argument("Unsupported MACE_Nonlinear e3nn Linear path.");
        const int width = 2*in.l+1;
        if (samples > std::numeric_limits<int>::max()/width)
            throw std::invalid_argument("MACE_Nonlinear linear batch is too large.");
        const int columns = samples*width;
        workspace.packed_input.resize(static_cast<std::size_t>(in.multiplicity)*columns);
        workspace.packed_output.assign(static_cast<std::size_t>(out.multiplicity)*columns, 0.0);
        for (int channel=0; channel<in.multiplicity; ++channel)
            for (int sample=0; sample<samples; ++sample)
                std::copy_n(
                    input_values.data()+sample*input.dimension()+in.offset+channel*width,
                    width,
                    workspace.packed_input.data()+channel*columns+sample*width);
        if (columns > 0)
            cblas_dgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                out.multiplicity, columns, in.multiplicity,
                instruction.path_weight,
                weights.data()+instruction.weight_offset, out.multiplicity,
                workspace.packed_input.data(), columns,
                0.0,
                workspace.packed_output.data(), columns);
        for (int channel=0; channel<out.multiplicity; ++channel)
            for (int sample=0; sample<samples; ++sample)
                for (int component=0; component<width; ++component)
                    output_values[sample*output.dimension()+out.offset+channel*width+component]
                        += workspace.packed_output[channel*columns+sample*width+component];
    }
    for (int sample=0; sample<samples; ++sample)
        for (int index=0; index<output.dimension(); ++index) {
            const std::size_t offset = static_cast<std::size_t>(sample)*output.dimension()+index;
            output_values[offset] = (output_values[offset]
                +(bias.empty() ? 0.0 : bias[index]))*output_mask[index];
        }
}

void E3Linear::reverse_batch(
    const std::vector<double>& output_adjoint,
    int samples,
    std::vector<double>& input_adjoint,
    E3LinearBatchWorkspace& workspace) const
{
    validate_batch_size(output_adjoint, samples, output.dimension(), "linear batch output adjoint");
    const std::size_t input_size = static_cast<std::size_t>(samples)*input.dimension();
    if (input_adjoint.size() != input_size)
        input_adjoint.assign(input_size, 0.0);
    for (const auto& instruction : instructions) {
        const auto& in = input.blocks[instruction.input_1];
        const auto& out = output.blocks[instruction.output];
        const int width = 2*in.l+1;
        if (samples > std::numeric_limits<int>::max()/width)
            throw std::invalid_argument("MACE_Nonlinear linear batch is too large.");
        const int columns = samples*width;
        workspace.packed_output.resize(static_cast<std::size_t>(out.multiplicity)*columns);
        workspace.packed_input.assign(static_cast<std::size_t>(in.multiplicity)*columns, 0.0);
        for (int channel=0; channel<out.multiplicity; ++channel)
            for (int sample=0; sample<samples; ++sample)
                for (int component=0; component<width; ++component) {
                    const int output_index = out.offset+channel*width+component;
                    workspace.packed_output[channel*columns+sample*width+component]
                        = output_mask[output_index]
                        *output_adjoint[sample*output.dimension()+output_index];
                }
        if (columns > 0)
            cblas_dgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                in.multiplicity, columns, out.multiplicity,
                instruction.path_weight,
                weights.data()+instruction.weight_offset, out.multiplicity,
                workspace.packed_output.data(), columns,
                0.0,
                workspace.packed_input.data(), columns);
        for (int channel=0; channel<in.multiplicity; ++channel)
            for (int sample=0; sample<samples; ++sample)
                for (int component=0; component<width; ++component)
                    input_adjoint[sample*input.dimension()+in.offset+channel*width+component]
                        += workspace.packed_input[channel*columns+sample*width+component];
    }
}

E3TensorProduct::E3TensorProduct(const nlohmann::json& data)
    : input_1(data.at("irreps_in1").get<std::string>()),
      input_2(data.at("irreps_in2").get<std::string>()),
      output(data.at("irreps_out").get<std::string>()),
      internal_weights(tensor_values(data.at("weight"))),
      output_mask(tensor_values(data.at("output_mask")))
{
    int weight_offset = 0;
    for (const auto& entry : data.at("instructions")) {
        auto instruction = parse_instruction(entry, true);
        if (instruction.input_1 < 0 || instruction.input_1 >= static_cast<int>(input_1.blocks.size())
            || instruction.input_2 < 0 || instruction.input_2 >= static_cast<int>(input_2.blocks.size())
            || instruction.output < 0 || instruction.output >= static_cast<int>(output.blocks.size()))
            throw std::invalid_argument("MACE_Nonlinear tensor-product instruction has an invalid irrep index.");
        const auto& in1 = input_1.blocks[instruction.input_1];
        const auto& in2 = input_2.blocks[instruction.input_2];
        const auto& out = output.blocks[instruction.output];
        const std::vector<int> expected_wigner_shape{
            2*in1.l+1, 2*in2.l+1, 2*out.l+1};
        if (instruction.wigner_shape != expected_wigner_shape
            || product(instruction.wigner_shape) != static_cast<int>(instruction.wigner_3j.size()))
            throw std::invalid_argument("MACE_Nonlinear tensor-product has an invalid Wigner tensor.");
        for (int a=0; a<expected_wigner_shape[0]; ++a)
            for (int b=0; b<expected_wigner_shape[1]; ++b)
                for (int c=0; c<expected_wigner_shape[2]; ++c) {
                    const double value = instruction.wigner_3j[
                        (a*expected_wigner_shape[1]+b)*expected_wigner_shape[2]+c];
                    if (value != 0.0)
                        instruction.nonzero_wigner.push_back({a,b,c,value});
                }
        if (instruction.mode == "uvu") {
            if (out.multiplicity != in1.multiplicity
                || instruction.path_shape != std::vector<int>{in1.multiplicity, in2.multiplicity})
                throw std::invalid_argument("MACE_Nonlinear uvu path has incompatible multiplicities or weight shape.");
        } else if (instruction.mode == "uuu") {
            if (out.multiplicity != in1.multiplicity
                || in2.multiplicity != in1.multiplicity
                || instruction.path_shape != std::vector<int>{in1.multiplicity})
                throw std::invalid_argument("MACE_Nonlinear uuu path has incompatible multiplicities or weight shape.");
        } else {
            throw std::invalid_argument(
                "Unsupported MACE_Nonlinear tensor-product connection mode: " + instruction.mode);
        }
        instruction.weight_offset = weight_offset;
        if (instruction.has_weight)
            weight_offset += product(instruction.path_shape);
        instructions.push_back(std::move(instruction));
    }
    weight_numel = weight_offset;
    validate_vector_size(output_mask, output.dimension(), "tensor-product output_mask");
    if (!internal_weights.empty())
        validate_vector_size(internal_weights, weight_numel, "tensor-product weight");
}

std::vector<double> E3TensorProduct::evaluate(
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& external_weights) const
{
    validate_vector_size(x1, input_1.dimension(), "tensor-product first input");
    validate_vector_size(x2, input_2.dimension(), "tensor-product second input");
    const auto& weights = external_weights.empty() ? internal_weights : external_weights;
    if (weight_numel != 0)
        validate_vector_size(weights, weight_numel, "tensor-product weights");
    std::vector<double> result(output.dimension(), 0.0);
    for (const auto& instruction : instructions) {
        const auto& in1 = input_1.blocks[instruction.input_1];
        const auto& in2 = input_2.blocks[instruction.input_2];
        const auto& out = output.blocks[instruction.output];
        const int d1 = 2*in1.l + 1;
        const int d2 = 2*in2.l + 1;
        const int d3 = 2*out.l + 1;
        for (int u=0; u<in1.multiplicity; ++u)
            for (int v=0; v<in2.multiplicity; ++v) {
                if (instruction.mode == "uuu" && u != v)
                    continue;
                const int w = u;
                const int weight_index = instruction.weight_offset
                    + (instruction.mode == "uuu" ? u : u*in2.multiplicity + v);
                const double path = instruction.has_weight
                    ? weights[weight_index]
                    : 1.0;
                const double scale = instruction.path_weight * path;
                for (const auto& entry : instruction.nonzero_wigner)
                    result[out.offset+w*d3+entry.c] += scale*entry.value
                        *x1[in1.offset+u*d1+entry.a]
                        *x2[in2.offset+v*d2+entry.b];
            }
    }
    for (int i=0; i<output.dimension(); ++i)
        result[i] *= output_mask[i];
    return result;
}

void E3TensorProduct::reverse(
    const std::vector<double>& x1,
    const std::vector<double>& x2,
    const std::vector<double>& external_weights,
    const std::vector<double>& output_adj,
    std::vector<double>& input_1_adj,
    std::vector<double>& input_2_adj,
    std::vector<double>& weights_adj) const
{
    validate_vector_size(output_adj, output.dimension(), "tensor-product output adjoint");
    const auto& weights = external_weights.empty() ? internal_weights : external_weights;
    if (weight_numel != 0)
        validate_vector_size(weights, weight_numel, "tensor-product weights");
    if (static_cast<int>(input_1_adj.size()) != input_1.dimension()) input_1_adj.assign(input_1.dimension(), 0.0);
    if (static_cast<int>(input_2_adj.size()) != input_2.dimension()) input_2_adj.assign(input_2.dimension(), 0.0);
    if (static_cast<int>(weights_adj.size()) != weight_numel) weights_adj.assign(weight_numel, 0.0);
    for (const auto& instruction : instructions) {
        const auto& in1 = input_1.blocks[instruction.input_1];
        const auto& in2 = input_2.blocks[instruction.input_2];
        const auto& out = output.blocks[instruction.output];
        const int d1 = 2*in1.l + 1, d2 = 2*in2.l + 1, d3 = 2*out.l + 1;
        for (int u=0; u<in1.multiplicity; ++u)
            for (int v=0; v<in2.multiplicity; ++v) {
                if (instruction.mode == "uuu" && u != v) continue;
                const int w = u;
                const int weight_index = instruction.weight_offset
                    + (instruction.mode == "uuu" ? u : u*in2.multiplicity + v);
                const double path = instruction.has_weight ? weights[weight_index] : 1.0;
                for (const auto& entry : instruction.nonzero_wigner) {
                    const double common = instruction.path_weight*entry.value
                        *output_mask[out.offset+w*d3+entry.c]
                        *output_adj[out.offset+w*d3+entry.c];
                    input_1_adj[in1.offset+u*d1+entry.a] += common*path
                        *x2[in2.offset+v*d2+entry.b];
                    input_2_adj[in2.offset+v*d2+entry.b] += common*path
                        *x1[in1.offset+u*d1+entry.a];
                    if (instruction.has_weight)
                        weights_adj[weight_index] += common
                            *x1[in1.offset+u*d1+entry.a]
                            *x2[in2.offset+v*d2+entry.b];
                }
            }
    }
}
