#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

struct AffineMLPBatchTape {
    int samples = 0;
    int dynamic_input_size = 0;
    std::vector<std::vector<double>> values;
};

struct AffineMLPBatchWorkspace {
    std::vector<double> adjoint;
    std::vector<double> scratch;
};

class AffineMLP {
public:
    AffineMLP() = default;
    explicit AffineMLP(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    bool supports_conditioned_input(int dynamic_input_size) const;
    std::vector<double> first_layer_contribution(
        int input_offset,
        const std::vector<double>& values) const;
    std::vector<double> evaluate(const std::vector<double>& input) const;
    std::vector<double> evaluate_conditioned(
        const std::vector<double>& input,
        const std::vector<double>& first_contribution,
        const std::vector<double>& second_contribution) const;
    void evaluate_conditioned_with_directional_derivative(
        const std::vector<double>& input,
        const std::vector<double>& input_derivative,
        const std::vector<double>& first_contribution,
        const std::vector<double>& second_contribution,
        std::vector<double>& output,
        std::vector<double>& output_derivative) const;
    std::vector<double> evaluate_gradient(
        const std::vector<double>& input,
        const std::vector<double>& output_adjoint) const;
    std::vector<double> evaluate_gradient_conditioned(
        const std::vector<double>& input,
        const std::vector<double>& first_contribution,
        const std::vector<double>& second_contribution,
        const std::vector<double>& output_adjoint) const;
    const std::vector<double>& evaluate_conditioned_batch(
        const std::vector<double>& input,
        int samples,
        int dynamic_input_size,
        const std::vector<double>& row_contributions,
        AffineMLPBatchTape& tape) const;
    void reverse_conditioned_batch(
        const std::vector<double>& output_adjoint,
        const AffineMLPBatchTape& tape,
        std::vector<double>& input_adjoint,
        AffineMLPBatchWorkspace& workspace) const;

private:
    struct Layer {
        enum class Type { Linear, LayerNorm, SiLU };
        Type type;
        int input_size = 0;
        int output_size = 0;
        double eps = 0.0;
        std::vector<double> weight;
        std::vector<double> bias;
    };

    std::vector<Layer> layers;
    std::vector<double> evaluate_impl(
        const std::vector<double>& input,
        const std::vector<double>* first_contribution,
        const std::vector<double>* second_contribution,
        std::vector<std::vector<double>>* tape) const;
    std::vector<double> evaluate_gradient_impl(
        const std::vector<double>& input,
        const std::vector<double>* first_contribution,
        const std::vector<double>* second_contribution,
        const std::vector<double>& output_adjoint) const;
};
