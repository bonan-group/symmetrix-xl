#include <algorithm>
#include <map>
#include <set>

#include "cblas.hpp"
#include "tools.hpp"

#include "multivariate_polynomial.hpp"

template <typename Precision>
MultivariatePolynomialT<Precision>::MultivariatePolynomialT(
    int num_variables,
    std::vector<Precision> coefficients,
    std::vector<std::vector<int>> monomials)
    : num_variables(num_variables),
      coefficients(coefficients),
      monomials(monomials)
{
    // TODO: sanitize input
    //      * check coefficients/monomials same size
    //      * ensure num_variables consistent with monomials
    //      * check for repeated monomials

    // comparison function governing lexographic ordering for monomial vectors
    auto lex_less = [](
            const std::vector<int>& v1, const std::vector<int>& v2) {
        if (v1.size() < v2.size()) {
            return true;
        } else if (v1.size() > v2.size()) {
            return false;
        } else {
            return v1 < v2;
        }
    };

    // store coefficients and monomials in lexographic order
    // (ordering ensured because the map sorts by key)
    std::map<std::vector<int>,Precision,decltype(lex_less)> map;
    for (int i=0; i<monomials.size(); ++i)
        map.insert({monomials[i], coefficients[i]});
    coefficients.clear();
    monomials.clear();
    for (const auto& [m,c] : map) {
        monomials.push_back(m);
        coefficients.push_back(c);
    }

    // create lexographically ordered node set from input
    std::set<std::vector<int>,decltype(lex_less)> node_set;
    for (int i=0; i<num_variables; ++i)
        node_set.insert({i});
    for (const auto& monomial : monomials)
        node_set.insert(monomial);

    // add auxiliary nodes until all nodes have two upstream factors
    num_auxiliary_nodes = 0;
    auto find_parents = [](const std::vector<int>& node,
                           const std::set<std::vector<int>,decltype(lex_less)>& node_set) {
        auto partitions = _two_part_partitions(node);
        for (const auto& partition : partitions)
            if (node_set.contains(partition[0]) && node_set.contains(partition[1]))
                return partition;
        return std::vector<std::vector<int>>();
    };
    for (auto node : node_set) {
        if (node.size() == 1)
            continue;
        auto factors = find_parents(node, node_set);
        while (factors.size() == 0) {
            node.pop_back();
            node_set.insert(node);
            num_auxiliary_nodes += 1;
            factors = find_parents(node, node_set);
        }
    }
    nodes = std::vector<std::vector<int>>(node_set.begin(), node_set.end());

    // find edges
    for (auto node : node_set) {
        if (node.size() == 1)
            continue;
        const auto factors = find_parents(node, node_set);
        const int i0 = std::distance(
            nodes.begin(),
            std::find(nodes.begin(), nodes.end(), factors[0]));
        const int i1 = std::distance(
            nodes.begin(),
            std::find(nodes.begin(), nodes.end(), factors[1]));
        edges.push_back({i0, i1});
    }

    // initialize node coefficients, values, and adjoints
    node_coefficients = std::vector<Precision>(nodes.size(), 0.0);
    int j = 0;
    for (int i=0; i<coefficients.size(); ++i) {
        while (monomials[i] != nodes[j]) {
            j += 1;
        }
        node_coefficients[j] = coefficients[i];
    }
    node_values = std::vector<Precision>(nodes.size());
    node_adjoints = std::vector<Precision>(nodes.size());
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate(
    const std::vector<Precision>& x)
    -> Precision
{
    initialize_forward_pass(x);
    forward_pass();
    return symmetrix_blas_dot<Precision>(node_coefficients.size(),
                      node_coefficients.data(), 1,
                      node_values.data(), 1);
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate_gradient(
    const std::vector<Precision>& x)
    -> std::tuple<Precision,std::vector<Precision>>
{
    initialize_forward_pass(x);
    forward_pass();
    auto f = symmetrix_blas_dot<Precision>(node_coefficients.size(),
                        node_coefficients.data(), 1,
                        node_values.data(), 1);
    initialize_backward_pass();
    backward_pass();
    auto g = extract_gradient_from_graph();
    return {f, g};
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate_gradient_directional(
    const std::vector<Precision>& x,
    const std::vector<Precision>& x_dot)
    -> std::tuple<Precision,std::vector<Precision>,std::vector<Precision>>
{
    initialize_forward_pass(x);
    auto node_value_dots = std::vector<Precision>(nodes.size(), 0.0);
    for (int i=0; i<num_variables; ++i)
        node_value_dots[i] = x_dot[i];

    for (int i=0; i<edges.size(); ++i) {
        const auto [i0, i1] = edges[i];
        const int output_node = num_variables+i;
        node_values[output_node] = node_values[i0] * node_values[i1];
        node_value_dots[output_node] =
            node_value_dots[i0] * node_values[i1]
            + node_values[i0] * node_value_dots[i1];
    }

    auto f = symmetrix_blas_dot<Precision>(node_coefficients.size(),
                        node_coefficients.data(), 1,
                        node_values.data(), 1);

    initialize_backward_pass();
    auto node_adjoints_dots = std::vector<Precision>(nodes.size(), 0.0);
    for (int i=edges.size()-1; i>=0; --i) {
        const auto [i0, i1] = edges[i];
        const int output_node = num_variables+i;
        node_adjoints[i0] += node_adjoints[output_node]*node_values[i1];
        node_adjoints[i1] += node_adjoints[output_node]*node_values[i0];
        node_adjoints_dots[i0] +=
            node_adjoints_dots[output_node]*node_values[i1]
            + node_adjoints[output_node]*node_value_dots[i1];
        node_adjoints_dots[i1] +=
            node_adjoints_dots[output_node]*node_values[i0]
            + node_adjoints[output_node]*node_value_dots[i0];
    }

    auto g = extract_gradient_from_graph();
    auto g_dot = std::vector<Precision>(num_variables, 0.0);
    for (int i=0; i<num_variables; ++i)
        g_dot[i] = node_adjoints_dots[i];
    return {f, g, g_dot};
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate_batch(
    const std::vector<Precision>& x,
    const int batch_size)
    -> std::tuple<std::vector<Precision>,std::vector<Precision>>
{
    batched_initialize_forward_pass(x, batch_size);
    batched_forward_pass(batch_size);
    auto f = std::vector<Precision>(batch_size, 0.0);
    for (int i=0; i<nodes.size(); ++i) {
        const Precision c = node_coefficients[i];
        for (int j=0; j<batch_size; ++j) {
            f[j] += c*node_values[i*batch_size+j];
        }
    }
    batched_initialize_backward_pass(batch_size);
    batched_backward_pass(batch_size);
    auto g = batched_extract_gradient_from_graph(batch_size);
    return {f, g};
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::initialize_forward_pass(
    const std::vector<Precision>& x)
{
    for (int i=0; i<num_variables; ++i)
        node_values[i] = x[i];
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::forward_pass()
{
    for (int i=0; i<edges.size(); ++i) {
        const auto [i0, i1] = edges[i];
        node_values[num_variables+i] = node_values[i0] * node_values[i1];
    }
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::initialize_backward_pass()
{
    symmetrix_blas_copy<Precision>(node_coefficients.size(),
                node_coefficients.data(), 1,
                node_adjoints.data(), 1);
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::backward_pass()
{
    for (int i=edges.size()-1; i>=0; --i) {
        const auto [i0, i1] = edges[i];
        node_adjoints[i0] += node_adjoints[num_variables+i]*node_values[i1];
        node_adjoints[i1] += node_adjoints[num_variables+i]*node_values[i0];
    }
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::extract_gradient_from_graph()
    -> std::vector<Precision>
{
    std::vector<Precision> g(num_variables, 0.0);
    for (int i=0; i<num_variables; ++i)
        g[i] = node_adjoints[i];
    return g;
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::batched_initialize_forward_pass(
    const std::vector<Precision>& x,
    const int batch_size)
{
    node_values.resize(batch_size * nodes.size());
    for (int i=0; i<num_variables; ++i)
        for (int j=0; j<batch_size; ++j)
            node_values[i*batch_size+j] = x[j*num_variables+i];
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::batched_forward_pass(
    const int batch_size)
{
    for (int i=0; i<edges.size(); ++i) {
        const auto [i0, i1] = edges[i];
        Precision* node_val = node_values.data() + (num_variables+i)*batch_size;
        const Precision* node_val_0 = node_values.data() + i0*batch_size;
        const Precision* node_val_1 = node_values.data() + i1*batch_size;
        for (int j=0; j<batch_size; ++j) {
            node_val[j] = node_val_0[j] * node_val_1[j];
        }
    }
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::batched_initialize_backward_pass(
    const int batch_size)
{
    node_adjoints.resize(batch_size * nodes.size());
    for (int i=0; i<node_coefficients.size(); ++i) {
        for (int j=0; j<batch_size; ++j) {
            node_adjoints[i*batch_size+j] = node_coefficients[i];
        }
    }
}

template <typename Precision>
void MultivariatePolynomialT<Precision>::batched_backward_pass(
    const int batch_size)
{
    for (int i=edges.size()-1; i>=0; --i) {
        const auto [i0, i1] = edges[i];
        Precision* node_adj = node_adjoints.data() + (num_variables+i)*batch_size;
        Precision* node_val_0 = node_values.data() + i0*batch_size;
        Precision* node_val_1 = node_values.data() + i1*batch_size;
        Precision* node_adj_0 = node_adjoints.data() + i0*batch_size;
        Precision* node_adj_1 = node_adjoints.data() + i1*batch_size;
        for (int j=0; j<batch_size; ++j)
            node_adj_0[j] += node_adj[j] * node_val_1[j];
        for (int j=0; j<batch_size; ++j)
            node_adj_1[j] += node_adj[j] * node_val_0[j];
    }
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::batched_extract_gradient_from_graph(
    const int batch_size)
    -> std::vector<Precision>
{
    std::vector<Precision> g(num_variables*batch_size, 0.0);
    for (int i=0; i<num_variables; ++i)
        for (int j=0; j<batch_size; ++j)
            g[j*num_variables+i] = node_adjoints[i*batch_size+j];
    return g;
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate_simple(const std::vector<Precision>& x) -> Precision
{
    Precision f = 0.0;
    for (int i=0; i<coefficients.size(); ++i) {
        Precision monomial = x[monomials[i][0]];
        for (int j=1; j<monomials[i].size(); ++j) {
            monomial *= x[monomials[i][j]];
        }
        f += coefficients[i] * monomial;
    }
    return f;
}

template <typename Precision>
auto MultivariatePolynomialT<Precision>::evaluate_gradient_simple(
    const std::vector<Precision>& x)
    -> std::tuple<Precision,std::vector<Precision>>
{
    Precision f = 0.0;
    for (int i=0; i<coefficients.size(); ++i) {
        Precision monomial = x[monomials[i][0]];
        for (int j=1; j<monomials[i].size(); ++j) {
            monomial *= x[monomials[i][j]];
        }
        f += coefficients[i] * monomial;
    }
    auto g = std::vector<Precision>(num_variables, 0.0);
    for (int i=0; i<coefficients.size(); ++i) {
        for (int j=0; j<monomials[i].size(); ++j) {
            Precision monomial_deriv = 1.0;
            for (int k=0; k<monomials[i].size(); ++k) {
                if (k==j) continue;
                monomial_deriv *= x[monomials[i][k]];
            }
            g[monomials[i][j]] += coefficients[i] * monomial_deriv;
        }
    }
    return {f,g};
}

template class MultivariatePolynomialT<float>;
template class MultivariatePolynomialT<double>;
