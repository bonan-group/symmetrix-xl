#pragma once

#include <array>
#include <span>
#include <vector>

#include "e3nn.hpp"

struct E3ProductBasisBatchWorkspace {
    E3LinearBatchWorkspace linear;
    std::vector<double> contracted;
    std::vector<double> feature_major;
    std::vector<double> product_major;
    std::vector<double> feature_adjoint;
};

// Product-basis evaluation used by EquivariantProductBasisBlock.  The MACE
// schema stores the U tensors emitted by mace.tools.cg, allowing this class to
// evaluate arbitrary correlation order without Python or e3nn at runtime.
class E3ProductBasis {
public:
    struct Tensor {
        std::vector<int> shape;
        std::vector<double> values;
        int offset(const std::vector<int>& index) const;
    };

    struct CompiledTerm {
        int degree = 0;
        std::array<int,3> indices{};
        std::vector<double> coefficients;
    };

    struct CompiledBlock {
        int angular_offset = 0;
        int width = 0;
        int num_elements = 0;
        std::vector<int> component_offsets;
        std::vector<CompiledTerm> terms;
    };

    explicit E3ProductBasis(const nlohmann::json& data);

    int input_dimension() const { return input.dimension(); }
    int output_dimension() const { return output.dimension(); }
    bool uses_compiled_plan() const { return !compiled_blocks.empty(); }
    int compiled_term_count() const;
    const std::vector<CompiledBlock>& compiled_plan() const { return compiled_blocks; }
    std::vector<double> evaluate(
        const std::vector<double>& node_features,
        const std::vector<double>& skip_connection,
        int element) const;
    void reverse(
        const std::vector<double>& node_features,
        int element,
        const std::vector<double>& output_adjoint,
        std::vector<double>& node_features_adjoint,
        std::vector<double>& skip_connection_adjoint) const;
    void evaluate_batch(
        const std::vector<double>& node_features,
        const std::vector<double>& skip_connections,
        const std::vector<int>& elements,
        int samples,
        std::vector<double>& output_values,
        E3ProductBasisBatchWorkspace& workspace) const;
    void reverse_batch(
        const std::vector<double>& node_features,
        const std::vector<int>& elements,
        const std::vector<double>& output_adjoint,
        int samples,
        std::vector<double>& node_features_adjoint,
        std::vector<double>& skip_connection_adjoint,
        E3ProductBasisBatchWorkspace& workspace) const;

private:
    struct Contraction {
        int correlation = 0;
        std::vector<Tensor> weights;
        Tensor weights_max;
        std::vector<Tensor> u_tensors;
    };

    void make_feature_major(
        std::span<const double> node_features,
        std::vector<double>& feature_major) const;
    void make_irrep_major(
        const std::vector<double>& feature_major,
        std::span<double> result) const;
    double evaluate_term(
        const Tensor& u,
        const Tensor& weights,
        const std::vector<double>& feature_major,
        int output_component,
        int feature,
        int element) const;
    void reverse_term(
        const Tensor& u,
        const Tensor& weights,
        const std::vector<double>& feature_major,
        int output_component,
        int feature,
        int element,
        double output_adjoint,
        std::vector<double>& feature_major_adjoint) const;
    bool has_mh1_product_layout(const nlohmann::json& data) const;
    void compile_mh1_product();
    void evaluate_compiled(
        const std::vector<double>& feature_major,
        int element,
        std::vector<double>& product_major) const;
    void reverse_compiled(
        const std::vector<double>& feature_major,
        int element,
        std::span<const double> contracted_adjoint,
        std::vector<double>& feature_major_adjoint) const;
    void evaluate_contraction(
        std::span<const double> node_features,
        int element,
        std::span<double> contracted,
        std::vector<double>& feature_major,
        std::vector<double>& product_major) const;
    void reverse_contraction(
        std::span<const double> node_features,
        int element,
        std::span<const double> contracted_adjoint,
        std::span<double> node_features_adjoint,
        std::vector<double>& feature_major,
        std::vector<double>& feature_major_adjoint) const;

    Irreps input;
    Irreps output;
    E3Linear linear;
    bool use_sc;
    int num_features;
    int angular_dimension;
    std::vector<Contraction> contractions;
    std::vector<CompiledBlock> compiled_blocks;
};
