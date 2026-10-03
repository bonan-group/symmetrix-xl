#pragma once

// Batched host GEMMs for the second-layer linear H2 = H1[:, 0] W_type + M1 W
// on a single host worker. The per-node GEMV formulation exists to spread
// nodes across OpenMP workers; with one worker, one GEMM for the shared M1
// weights and one gathered GEMM per node type use the BLAS far better.

#include <cstddef>
#include <vector>

#include "cblas.hpp"

namespace symmetrix {

// Node indices grouped by raw node type.
inline std::vector<std::vector<int>> host_nodes_by_type(
    const int* node_types, const int num_nodes)
{
    std::vector<std::vector<int>> groups;
    for (int node=0; node<num_nodes; ++node) {
        const int type = node_types[node];
        if (type >= static_cast<int>(groups.size()))
            groups.resize(type+1);
        groups[type].push_back(node);
    }
    return groups;
}

// H2[node, :] = H1[node, 0, :] W_H1[type] + M1[node, :] W_M1, row-major with
// the given node pitches; weight matrices are [channels, channels].
template <typename Precision>
void host_batched_h2_forward(
    const int num_nodes, const int channels, const int* node_types,
    const Precision* h1, const std::size_t h1_pitch,
    const Precision* m1, const std::size_t m1_pitch,
    const Precision* h1_weights, const std::size_t h1_weight_pitch,
    const Precision* m1_weights,
    double* h2, const std::size_t h2_pitch)
{
    const std::size_t width = static_cast<std::size_t>(channels);
    std::vector<Precision> output(static_cast<std::size_t>(num_nodes)*width);
    symmetrix_blas_gemm<Precision>(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        num_nodes, channels, channels,
        Precision(1), m1, static_cast<int>(m1_pitch), m1_weights, channels,
        Precision(0), output.data(), channels);
    std::vector<Precision> rows;
    std::vector<Precision> products;
    const auto groups = host_nodes_by_type(node_types, num_nodes);
    for (std::size_t type=0; type<groups.size(); ++type) {
        const auto& nodes = groups[type];
        if (nodes.empty())
            continue;
        rows.resize(nodes.size()*width);
        products.resize(nodes.size()*width);
        for (std::size_t row=0; row<nodes.size(); ++row)
            for (std::size_t channel=0; channel<width; ++channel)
                rows[row*width+channel] = h1[nodes[row]*h1_pitch+channel];
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor, CblasNoTrans, CblasNoTrans,
            static_cast<int>(nodes.size()), channels, channels,
            Precision(1), rows.data(), channels,
            h1_weights+type*h1_weight_pitch, channels,
            Precision(0), products.data(), channels);
        for (std::size_t row=0; row<nodes.size(); ++row)
            for (std::size_t channel=0; channel<width; ++channel)
                output[nodes[row]*width+channel] += products[row*width+channel];
    }
    for (std::size_t node=0; node<static_cast<std::size_t>(num_nodes); ++node)
        for (std::size_t channel=0; channel<width; ++channel)
            h2[node*h2_pitch+channel] =
                static_cast<double>(output[node*width+channel]);
}

// Adjoint of host_batched_h2_forward: M1_adj = H2_adj W_M1^T is written and
// H1_adj[node, 0, :] += H2_adj W_H1[type]^T is accumulated.
template <typename Precision>
void host_batched_h2_reverse(
    const int num_nodes, const int channels, const int* node_types,
    const double* h2_adjoint, const std::size_t h2_pitch,
    const Precision* h1_weights, const std::size_t h1_weight_pitch,
    const Precision* m1_weights,
    Precision* h1_adjoint, const std::size_t h1_pitch,
    Precision* m1_adjoint, const std::size_t m1_pitch)
{
    const std::size_t width = static_cast<std::size_t>(channels);
    std::vector<Precision> adjoint(static_cast<std::size_t>(num_nodes)*width);
    for (std::size_t node=0; node<static_cast<std::size_t>(num_nodes); ++node)
        for (std::size_t channel=0; channel<width; ++channel)
            adjoint[node*width+channel] =
                static_cast<Precision>(h2_adjoint[node*h2_pitch+channel]);
    symmetrix_blas_gemm<Precision>(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        num_nodes, channels, channels,
        Precision(1), adjoint.data(), channels, m1_weights, channels,
        Precision(0), m1_adjoint, static_cast<int>(m1_pitch));
    std::vector<Precision> rows;
    std::vector<Precision> products;
    const auto groups = host_nodes_by_type(node_types, num_nodes);
    for (std::size_t type=0; type<groups.size(); ++type) {
        const auto& nodes = groups[type];
        if (nodes.empty())
            continue;
        rows.resize(nodes.size()*width);
        products.resize(nodes.size()*width);
        for (std::size_t row=0; row<nodes.size(); ++row)
            for (std::size_t channel=0; channel<width; ++channel)
                rows[row*width+channel] = adjoint[nodes[row]*width+channel];
        symmetrix_blas_gemm<Precision>(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            static_cast<int>(nodes.size()), channels, channels,
            Precision(1), rows.data(), channels,
            h1_weights+type*h1_weight_pitch, channels,
            Precision(0), products.data(), channels);
        for (std::size_t row=0; row<nodes.size(); ++row)
            for (std::size_t channel=0; channel<width; ++channel)
                h1_adjoint[nodes[row]*h1_pitch+channel] +=
                    products[row*width+channel];
    }
}

}  // namespace symmetrix
