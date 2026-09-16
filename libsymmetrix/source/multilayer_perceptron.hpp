#pragma once

#include <tuple>
#include <vector>

template <typename Precision>
class MultilayerPerceptronT {

public:

MultilayerPerceptronT();
MultilayerPerceptronT(
    std::vector<int> shape,
    std::vector<std::vector<Precision>> weights,
    Precision activation_scale_factor = 1.0);

auto evaluate(std::vector<Precision> input) -> std::vector<Precision>;
auto evaluate_gradient(std::vector<Precision> input) -> std::tuple<std::vector<Precision>,std::vector<Precision>>;
auto evaluate_gradient_directional(
        std::vector<Precision> input,
        std::vector<Precision> input_dot)
        -> std::tuple<std::vector<Precision>,std::vector<Precision>,std::vector<Precision>>;
auto evaluate_batch(std::vector<Precision> input, const int batch_size) -> std::vector<Precision>;
auto evaluate_penultimate_batch(std::vector<Precision> input, const int batch_size)
    -> std::vector<Precision>;
auto evaluate_gradient_batch(std::vector<Precision> input, const int batch_size) -> std::tuple<std::vector<Precision>,std::vector<Precision>>;

private:

std::vector<int> shape;
std::vector<std::vector<Precision>> weights;
Precision activation_scale_factor;

std::vector<std::vector<Precision>> node_values;
std::vector<std::vector<Precision>> node_derivs;
std::vector<std::vector<Precision>> node_activation_derivs;

};

using MultilayerPerceptron = MultilayerPerceptronT<double>;

extern template class MultilayerPerceptronT<float>;
extern template class MultilayerPerceptronT<double>;
