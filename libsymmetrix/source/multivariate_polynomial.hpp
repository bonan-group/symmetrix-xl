#pragma once

#include <array>
#include <tuple>
#include <vector>

template <typename Precision>
class MultivariatePolynomialT
{

public:

MultivariatePolynomialT(int num_variables,
                        std::vector<Precision> coefficients,
                        std::vector<std::vector<int>> monomials);

    auto evaluate(const std::vector<Precision>& x) -> Precision;
    auto evaluate_gradient(const std::vector<Precision>& x) -> std::tuple<Precision,std::vector<Precision>>;
    auto evaluate_gradient_directional(
        const std::vector<Precision>& x,
        const std::vector<Precision>& x_dot)
        -> std::tuple<Precision,std::vector<Precision>,std::vector<Precision>>;
    auto evaluate_batch(const std::vector<Precision>& x, const int batch_size) -> std::tuple<std::vector<Precision>,std::vector<Precision>>;

// polynomial specification
int num_variables;
std::vector<Precision> coefficients;
std::vector<std::vector<int>> monomials;

// graph-related variables
int num_auxiliary_nodes;
std::vector<std::vector<int>> nodes;
std::vector<std::array<int,2>> edges;
std::vector<Precision> node_coefficients;
std::vector<Precision> node_values;
std::vector<Precision> node_adjoints;

// used during recursive evaluation
void initialize_forward_pass(const std::vector<Precision>& x);
void forward_pass();
void initialize_backward_pass();
void backward_pass();
auto extract_gradient_from_graph() -> std::vector<Precision>;

// used during recursive evaluation with batching
void batched_initialize_forward_pass(const std::vector<Precision>& x, const int batch_size);
void batched_forward_pass(const int batch_size);
void batched_initialize_backward_pass(const int batch_size);
void batched_backward_pass(const int batch_size);
auto batched_extract_gradient_from_graph(const int batch_size) -> std::vector<Precision>;

// non-recursive evaluation, used for testing
auto evaluate_simple(const std::vector<Precision>& x) -> Precision;
auto evaluate_gradient_simple(const std::vector<Precision>& x) -> std::tuple<Precision,std::vector<Precision>>;

};

using MultivariatePolynomial = MultivariatePolynomialT<double>;

extern template class MultivariatePolynomialT<float>;
extern template class MultivariatePolynomialT<double>;
